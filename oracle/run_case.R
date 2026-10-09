#!/usr/bin/env Rscript
# Public black-box API only. Do not print, inspect, or resolve function bodies.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop('Usage: Rscript oracle/run_case.R input.json [output.json]')
input <- fromJSON(args[[1]], simplifyVector = FALSE)
`%or%` <- function(x, y) if (is.null(x)) y else x
plain_matrix <- function(x) unname(as.matrix(x))
public_family <- function(name, link = NULL, extended = FALSE) {
  if (name %in% c('gaussian', 'binomial', 'poisson', 'Gamma')) {
    fun <- getExportedValue('stats', name)
  } else if (extended && name %in% c('nbinom1', 'nbinom2', 'beta')) {
    fun <- getExportedValue('glmmTMB', if (name == 'beta') 'beta_family' else name)
  } else stop('Unsupported family')
  if (is.null(link)) fun() else fun(link = link)
}
prediction_result <- function(value) {
  if (is.list(value) && !is.null(value$fit)) {
    list(prediction = if (is.matrix(value$fit)) plain_matrix(value$fit) else unname(value$fit),
         se_fit = if (is.matrix(value$se.fit)) plain_matrix(value$se.fit) else unname(value$se.fit),
         columns = if (is.matrix(value$fit)) colnames(value$fit) else NULL)
  } else {
    list(prediction = if (is.matrix(value)) plain_matrix(value) else unname(value),
         columns = if (is.matrix(value)) colnames(value) else NULL)
  }
}
display_summary <- function(m) {
  # Canonicalize only call provenance, using the object's public call field.
  cl <- stats::getCall(m)
  if (!is.null(cl$data)) cl$data <- quote(data)
  if (!is.null(cl$control)) cl$control <- quote(control)
  if (!is.null(cl$weights)) cl$weights <- quote(weights)
  if (!is.null(cl$offset)) cl$offset <- quote(offset)
  if (inherits(m, 'merMod')) m@call <- cl else m$call <- cl
  paste(capture.output(summary(m)), collapse = '\n')
}
table_result <- function(x) {
  d <- as.data.frame(x)
  list(rows = rownames(d), columns = names(d), values = lapply(d, function(z) {
    if (is.factor(z)) as.character(z) else unname(z)
  }))
}
make_data <- function(s) {
  if (!is.null(s$dataset)) {
    e <- new.env()
    data(list = s$dataset, package = s$package, envir = e)
    d <- get(s$dataset, envir = e)
    if (!is.null(s$cache_csv)) write.csv(d, s$cache_csv, row.names = FALSE)
  } else if (!is.null(s$csv)) {
    d <- read.csv(s$csv, stringsAsFactors = FALSE)
  } else {
    d <- as.data.frame(lapply(s$data, function(z) unlist(lapply(z, function(v) if (is.null(v)) NA else v))), stringsAsFactors = FALSE)
  }
  for (nm in unlist(s$factors %or% list())) d[[nm]] <- factor(d[[nm]])
  d
}
structure_arg <- function(s, kind) {
  if (is.null(s)) return(NULL)
  constructors <- if (kind == 'correlation') c('corAR1','corCompSymm','corSymm','corARMA') else c('varIdent','varPower','varExp')
  if (!s$name %in% constructors) stop('Unapproved covariance constructor')
  a <- s$args %or% list()
  if (!is.null(a$form)) a$form <- as.formula(a$form)
  if (!is.null(a$value)) a$value <- unlist(a$value)
  do.call(getExportedValue('nlme', s$name), a)
}
fit_model <- function(s, d) {
  form <- as.formula(s$formula)
  a <- s$args %or% list()
  if (!is.null(s$weights) && is.character(s$weights)) a$weights <- d[[s$weights]]
  if (!is.null(s$offset)) a$offset <- d[[s$offset]]
  if (s$call == 'lmer') {
    a$REML <- a$REML %or% TRUE
    a$control <- lme4::lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 200000, rhoend = 1e-10))
    do.call(lmerTest::lmer, c(list(formula = form, data = d), a))
  } else if (s$call == 'glmer') {
    fam <- s$family %or% 'binomial'
    a$family <- if (fam == 'binomial') binomial(link = s$link %or% 'logit') else if (fam == 'poisson') poisson(link = 'log') else stop('Unsupported family')
    a$nAGQ <- a$nAGQ %or% 1L
    a$control <- lme4::glmerControl(optimizer = 'bobyqa', tolPwrss = s$glmer_tol %or% 1e-14, nAGQ0initStep = s$glmer_nAGQ0initStep %or% TRUE, compDev = s$glmer_compDev %or% TRUE, optCtrl = list(maxfun = 200000, rhoend = 1e-10))
    do.call(lme4::glmer, c(list(formula = form, data = d), a))
  } else if (s$call == 'gls') {
    a$method <- a$method %or% 'REML'
    a$correlation <- structure_arg(s$correlation, 'correlation')
    a$weights <- structure_arg(s$variance, 'variance')
    controls <- modifyList(list(msMaxIter = 1000L, tolerance = 1e-12, msTol = 1e-14, returnObject = TRUE, opt = 'nlminb', .relStep = 1e-4), s$gls_control %or% list())
    a$control <- do.call(nlme::glsControl, controls)
    do.call(nlme::gls, c(list(model = form, data = d), a))
  } else if (s$call == 'lm') {
    do.call(stats::lm, c(list(formula = form, data = d), a))
  } else if (s$call == 'glm') {
    a$family <- if ((s$family %or% 'poisson') == 'binomial') binomial(link = s$link %or% 'logit') else poisson(link = 'log')
    a$control <- stats::glm.control(epsilon = 1e-12, maxit = 1000L)
    do.call(stats::glm, c(list(formula = form, data = d), a))
  } else if (s$call == 'gam') {
    a$family <- public_family(s$family %or% 'gaussian', s$link)
    a$method <- a$method %or% 'GCV.Cp'
    a$control <- do.call(mgcv::gam.control, modifyList(list(epsilon = 1e-10, maxit = 1000L), s$gam_control %or% list()))
    if (!is.null(a$sp)) a$sp <- unlist(a$sp)
    if (!is.null(a$knots)) a$knots <- lapply(a$knots, unlist)
    do.call(mgcv::gam, c(list(formula = form, data = d), a))
  } else if (s$call == 'glmmTMB') {
    a$family <- public_family(s$family %or% 'gaussian', s$link, extended = TRUE)
    a$ziformula <- as.formula(s$ziformula %or% '~ 0')
    a$dispformula <- as.formula(s$dispformula %or% '~ 1')
    a$control <- do.call(glmmTMB::glmmTMBControl, modifyList(list(optCtrl = list(iter.max = 10000L, eval.max = 10000L, rel.tol = 1e-10), parallel = 1L), s$tmb_control %or% list()))
    do.call(glmmTMB::glmmTMB, c(list(formula = form, data = d), a))
  } else stop('Unapproved model function')
}
extract_gam <- function(m, s) {
  sm <- summary(m)
  ans <- list(beta = unname(coef(m)), coef_names = names(coef(m)),
              vcov = plain_matrix(vcov(m)), logLik = as.numeric(logLik(m)),
              AIC = AIC(m), BIC = BIC(m), fitted = unname(fitted(m)),
              residuals = unname(residuals(m)), summary = display_summary(m),
              coefficients = table_result(sm$p.table), smooth_table = table_result(sm$s.table),
              edf = unname(m$edf), edf1 = unname(m$edf1), sp = unname(m$sp),
              sp_names = names(m$sp), scale = unname(sm$scale),
              df_resid = df.residual(m), deviance = deviance(m),
              method = m$method, criterion = unname(m$gcv.ubre),
              rank = m$rank, converged = m$converged,
              X = plain_matrix(predict(m, type = 'lpmatrix')),
              linear_predictor = unname(predict(m, type = 'link')),
              smooths = lapply(m$smooth, function(z) {
                list(label = z$label, term = z$term, by = z$by,
                     first_para = z$first.para, last_para = z$last.para,
                     rank = unname(z$rank), null_space_dim = z$null.space.dim,
                     S = lapply(z$S, plain_matrix), S_scale = unname(z$S.scale),
                     first_sp = z$first.sp, last_sp = z$last.sp,
                     F = if (isTRUE(s$smooth_basis_diagnostics)) unname(z$F) else NULL,
                     xp = if (isTRUE(s$smooth_basis_diagnostics)) unname(z$xp) else NULL,
                     UZ = if (isTRUE(s$smooth_basis_diagnostics) && is.matrix(z$UZ)) plain_matrix(z$UZ) else NULL,
                     Xu = if (isTRUE(s$smooth_basis_diagnostics) && is.matrix(z$Xu)) plain_matrix(z$Xu) else NULL,
                     shift = if (isTRUE(s$smooth_basis_diagnostics)) unname(z$shift) else NULL,
                     cmX = if (isTRUE(s$smooth_basis_diagnostics)) unname(z$cmX) else NULL)
              }))
  if (!is.null(m$outer.info)) {
    ans$optimizer_diagnostics <- list(convergence = m$outer.info$conv,
                                     iterations = m$outer.info$iter,
                                     gradient = unname(m$outer.info$grad),
                                     hessian = if (is.matrix(m$outer.info$hess)) plain_matrix(m$outer.info$hess) else NULL)
  }
  if (isTRUE(s$optimizer_diagnostics)) {
    ans$optimizer_diagnostics <- c(ans$optimizer_diagnostics, list(weights = unname(m$weights), prior_weights = unname(m$prior.weights), working_weights = unname(m$working.weights), Vp = plain_matrix(m$Vp), Ve = plain_matrix(m$Ve), R = if (is.matrix(m$R)) plain_matrix(m$R) else NULL, deviance = m$deviance, df_residual = m$df.residual, scale = m$sig2))
  }
  if (isTRUE(s$gam_check)) ans$gam_check <- extract_gam_check(m, s)
  if (isTRUE(s$gam_prediction)) {
    nd <- if (is.null(s$newdata)) NULL else make_data(list(data = s$newdata, factors = s$factors))
    ans$prediction <- prediction_result(predict(m, newdata = nd, se.fit = TRUE, type = 'link'))
    ans$terms <- prediction_result(predict(m, newdata = nd, se.fit = TRUE, type = 'terms'))
  }
  ans
}
extract_gam_check <- function(m, s) {
  set.seed(s$diagnostic_seed %or% 1L)
  list(k_check = table_result(mgcv::k.check(m, subsample = s$k_subsample %or% 5000L,
                                          n.rep = s$k_rep %or% 400L)),
       converged = m$converged, rank = m$rank, model_rank = length(coef(m)),
       scale = m$sig2, deviance = deviance(m), residual_df = df.residual(m))
}
extract_tmb <- function(m, s) {
  b <- glmmTMB::fixef(m)
  vv <- vcov(m)
  sm <- summary(m)
  rr <- glmmTMB::ranef(m, condVar = TRUE)
  vc <- glmmTMB::VarCorr(m)
  list(beta = unname(b$cond), coef_names = names(b$cond), vcov = plain_matrix(vv$cond),
       fixef = lapply(b, unname), fixef_names = lapply(b, names),
       vcov_components = lapply(vv, plain_matrix),
       vcov_full = plain_matrix(vcov(m, full = TRUE)),
       vcov_full_names = colnames(vcov(m, full = TRUE)),
       logLik = as.numeric(logLik(m)), AIC = AIC(m), BIC = BIC(m),
       fitted = unname(fitted(m)), residuals = unname(residuals(m)),
       summary = display_summary(m), sigma = sigma(m), df_resid = df.residual(m),
       coefficients = lapply(sm$coefficients, table_result),
       ranef = lapply(rr, function(component) lapply(component, function(z) {
         list(levels = rownames(z), names = names(z), values = plain_matrix(z),
              conditional_variance = attr(z, 'condVar'))
       })),
       VarCorr = lapply(vc[c('cond', 'zi')], function(component) lapply(component, function(z) {
         list(names = colnames(z), covariance = plain_matrix(z),
              sd = unname(attr(z, 'stddev')), correlation = plain_matrix(attr(z, 'correlation')))
       })),
       predictions = list(response = unname(predict(m, type = 'response')),
                          cond = unname(predict(m, type = 'conditional')),
                          zprob = unname(predict(m, type = 'zprob')),
                          disp = unname(predict(m, type = 'disp'))),
       optimizer_diagnostics = list(convergence = m$fit$convergence,
                                    message = m$fit$message, iterations = m$fit$iterations,
                                    objective = unname(m$fit$objective),
                                    pdHess = m$sdr$pdHess))
}
extract_model <- function(m, s) {
  if (inherits(m, 'gam')) return(extract_gam(m, s))
  if (inherits(m, 'glmmTMB')) return(extract_tmb(m, s))
  mixed <- inherits(m, 'merMod')
  b <- if (mixed) lme4::fixef(m) else stats::coef(m)
  ans <- list(beta = unname(b), coef_names = names(b), vcov = plain_matrix(stats::vcov(m)), logLik = as.numeric(logLik(m)), AIC = AIC(m), BIC = BIC(m), fitted = unname(fitted(m)), residuals = unname(residuals(m)), summary = display_summary(m))
  if (mixed) {
    ans$theta <- unname(lme4::getME(m, 'theta'))
    if (s$call == 'glmer') {
      ans$response <- unname(lme4::getME(m, 'y'))
      ans$prior_weights <- unname(stats::weights(m, type = 'prior'))
      if (isTRUE(s$optimizer_diagnostics)) {
        ans$optimizer_diagnostics <- list(optimizer = m@optinfo$optimizer, derivatives = m@optinfo$derivs, convergence = m@optinfo$conv)
        ans$vcov_factor <- plain_matrix(stats::vcov(m, use.hessian = FALSE))
      }
    }
    ans$VarCorr <- table_result(as.data.frame(lme4::VarCorr(m)))
    ans$singular <- lme4::isSingular(m)
    rr <- lme4::ranef(m, condVar = TRUE)
    ans$ranef <- lapply(rr, function(z) list(levels = rownames(z), names = names(z), values = plain_matrix(z), conditional_variance = attr(z, 'postVar')))
    ans$deviance <- if (s$call == 'lmer') -2 * as.numeric(logLik(m)) else deviance(m)
    ans$sigma <- sigma(m)
    ans$coefficients <- table_result(coef(summary(m)))
  } else if (inherits(m, 'gls')) {
    ans$sigma <- m$sigma
    ans$apVar <- if (is.matrix(m$apVar)) plain_matrix(m$apVar) else m$apVar
    ans$apVar_parameters <- if (is.matrix(m$apVar)) unname(attr(m$apVar, 'Pars')) else NULL
    ans$df_resid <- nrow(m$data %or% make_data(s)) - length(b)
    ans$coefficients <- table_result(summary(m)$tTable)
    if (!is.null(m$modelStruct$corStruct)) ans$correlation <- unname(coef(m$modelStruct$corStruct, unconstrained = FALSE))
    if (!is.null(m$modelStruct$varStruct)) ans$variance <- coef(m$modelStruct$varStruct, unconstrained = FALSE)
    ans$intervals <- tryCatch(list(coef = plain_matrix(nlme::intervals(m)$coef), sigma = unname(nlme::intervals(m)$sigma)), error = function(e) list(error = conditionMessage(e)))
  } else {
    ans$df_resid <- df.residual(m)
    ans$coefficients <- table_result(coef(summary(m)))
    ans$deviance <- deviance(m)
  }
  ans
}
run_one <- function(s) {
  id <- s$id %or% 'unnamed'
  ws <- character()
  fitted_ll <- NULL
  fit_seconds <- NULL
  ans <- tryCatch(withCallingHandlers({
    if ((s$operation %or% '') == 'versions') {
      ps <- c('jsonlite','lme4','lmerTest','car','emmeans','pbkrtest','nlme','mgcv','glmmTMB','TMB')
      list(R = R.version.string, packages = setNames(lapply(ps, function(p) tryCatch(as.character(packageVersion(p)), error = function(e) NA_character_)), ps))
    } else if ((s$operation %or% '') == 'rng') {
      RNGkind(kind = 'Mersenne-Twister', normal.kind = 'Inversion', sample.kind = 'Rejection')
      seed <- as.integer(s$seed %or% 17L)
      count <- as.integer(s$n %or% 10L)
      set.seed(seed)
      initial_state <- .Random.seed
      uniforms <- runif(count)
      set.seed(seed)
      permutation <- sample.int(count, size = as.integer(s$size %or% count), replace = isTRUE(s$replace))
      list(initial_state = unname(initial_state), uniforms = unname(uniforms), permutation = unname(permutation), final_state = unname(.Random.seed))
    } else {
      options(contrasts = if ((s$contrasts %or% 'treatment') == 'sum') c('contr.sum','contr.poly') else c('contr.treatment','contr.poly'))
      d <- make_data(s)
      if ((s$operation %or% '') == 'data') {
        list(columns = names(d), data = lapply(d, function(z) if (is.factor(z)) as.character(z) else z))
      } else if ((s$operation %or% '') == 'compare') {
        fits <- lapply(s$models, function(ms) fit_model(ms, d))
        table_result(do.call(stats::anova, c(fits, list(refit = s$refit %or% TRUE))))
      } else {
        if (isTRUE(s$benchmark)) {
          fit_seconds <- unname(system.time(m <- fit_model(s, d))[['elapsed']])
        } else m <- fit_model(s, d)
        fitted_ll <- as.numeric(logLik(m))
        op <- s$operation %or% 'fit'
        if (op == 'fit') extract_model(m, s)
        else if (op == 'gam.check') {
          if (!inherits(m, 'gam')) stop('gam.check requires a GAM fit')
          extract_gam_check(m, s)
        }
        else if (op == 'partial_effects') {
          if (!inherits(m, 'gam')) stop('partial_effects requires a GAM fit')
          nd <- if (is.null(s$newdata)) d else make_data(list(data = s$newdata, factors = s$factors))
          a <- list(object = m, newdata = nd, type = 'terms', se.fit = TRUE)
          if (!is.null(s$terms)) a$terms <- unlist(s$terms)
          prediction_result(do.call(stats::predict, a))
        }
        else if (op == 'anova') {
          if (inherits(m, 'gls')) table_result(stats::anova(m, type = s$type %or% 'sequential', adjustSigma = s$adjust_sigma %or% TRUE))
          else table_result(stats::anova(m, type = s$type %or% 3L, ddf = s$ddf %or% 'Satterthwaite'))
        }
        else if (op == 'hypotheses') {
          tab <- stats::anova(m, type = s$type %or% 3L, ddf = 'Satterthwaite')
          lapply(lmerTest::show_tests(tab), plain_matrix)
        }
        else if (op == 'Anova') {
          a <- list(mod = m, type = s$type %or% 2L)
          if (!is.null(s$test)) a$test.statistic <- s$test
          if (inherits(m, 'glmmTMB')) a$component <- s$component %or% 'cond'
          table_result(do.call(car::Anova, a))
        } else if (op %in% c('emmeans','emtrends','joint_tests')) {
          a <- s$emm_args %or% list()
          if (is.character(a$specs) && startsWith(a$specs, '~')) a$specs <- as.formula(a$specs)
          if (!is.null(a$at)) a$at <- lapply(a$at, unlist)
          a$lmer.df <- s$ddf %or% 'kenward-roger'
          if (op == 'joint_tests') table_result(do.call(emmeans::joint_tests, c(list(object = m), a)))
          else {
            fun <- if (op == 'emmeans') emmeans::emmeans else emmeans::emtrends
            e <- do.call(fun, c(list(object = m), a))
            if (!is.null(s$contrast)) {
              cm <- s$contrast$method %or% 'pairwise'
              if (is.list(cm)) cm <- lapply(cm, unlist)
              e <- emmeans::contrast(e, method = cm, adjust = s$contrast$adjust %or% 'none')
            }
            table_result(summary(e, type = s$response_type %or% 'link', infer = c(TRUE, TRUE), adjust = s$adjust %or% if (!is.null(s$contrast)) s$contrast$adjust %or% 'none' else 'none'))
          }
        } else if (op == 'predict') {
          nd <- as.data.frame(lapply(s$newdata, function(z) unlist(lapply(z, function(v) if (is.null(v)) NA else v))), stringsAsFactors = FALSE)
          if (inherits(m, 'merMod')) list(prediction = unname(predict(m, newdata = nd, re.form = if (isTRUE(s$population)) NA else NULL, allow.new.levels = s$allow_new_levels %or% FALSE, type = s$prediction_type %or% 'response')))
          else if (inherits(m, 'gam')) {
            a <- list(object = m, newdata = nd, type = s$prediction_type %or% 'response',
                      se.fit = s$se_fit %or% FALSE, unconditional = s$unconditional %or% FALSE)
            if (!is.null(s$terms)) a$terms <- unlist(s$terms)
            prediction_result(do.call(stats::predict, a))
          } else if (inherits(m, 'glmmTMB')) {
            pt <- s$prediction_type %or% 'response'
            if (pt == 'cond') pt <- 'conditional'
            prediction_result(predict(m, newdata = nd, type = pt, se.fit = s$se_fit %or% FALSE,
                                      re.form = if (isTRUE(s$population)) NA else NULL,
                                      allow.new.levels = s$allow_new_levels %or% FALSE))
          } else list(prediction = unname(predict(m, newdata = nd)))
        } else stop('Unapproved extraction operation')
      }
    }
  }, warning = function(w) { ws <<- c(ws, conditionMessage(w)); invokeRestart('muffleWarning') }, message = function(m) { ws <<- c(ws, conditionMessage(m)); invokeRestart('muffleMessage') }), error = function(e) list(error = conditionMessage(e)))
  if (!is.null(fitted_ll) && is.null(ans$error)) ans$logLik <- fitted_ll
  if (!is.null(fit_seconds) && is.null(ans$error)) ans$fit_seconds <- fit_seconds
  list(id = id, result = ans, warnings = as.list(unique(ws)))
}
cases <- input$cases %or% list(input)
output <- lapply(cases, run_one)
js <- toJSON(output, auto_unbox = TRUE, digits = 16, null = 'null', na = 'null', pretty = TRUE, matrix = 'rowmajor')
if (length(args) > 1L) writeLines(js, args[[2]]) else cat(js, '\n')
