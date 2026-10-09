#!/usr/bin/env Rscript
# Public black-box API only. Do not print, inspect, or resolve function bodies.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop('Usage: Rscript oracle/run_case.R input.json [output.json]')
input <- fromJSON(args[[1]], simplifyVector = FALSE)
`%or%` <- function(x, y) if (is.null(x)) y else x
plain_matrix <- function(x) unname(as.matrix(x))
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
    d <- as.data.frame(lapply(s$data, function(z) unlist(z)), stringsAsFactors = FALSE)
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
    a$control <- lme4::glmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 200000, rhoend = 1e-10))
    do.call(lme4::glmer, c(list(formula = form, data = d), a))
  } else if (s$call == 'gls') {
    a$method <- a$method %or% 'REML'
    a$correlation <- structure_arg(s$correlation, 'correlation')
    a$weights <- structure_arg(s$variance, 'variance')
    a$control <- nlme::glsControl(msMaxIter = 300L, tolerance = 1e-9, msTol = 1e-9, opt = 'optim')
    do.call(nlme::gls, c(list(model = form, data = d), a))
  } else if (s$call == 'lm') {
    do.call(stats::lm, c(list(formula = form, data = d), a))
  } else if (s$call == 'glm') {
    a$family <- if ((s$family %or% 'poisson') == 'binomial') binomial(link = s$link %or% 'logit') else poisson(link = 'log')
    do.call(stats::glm, c(list(formula = form, data = d), a))
  } else stop('Unapproved model function')
}
extract_model <- function(m, s) {
  mixed <- inherits(m, 'merMod')
  b <- if (mixed) lme4::fixef(m) else stats::coef(m)
  ans <- list(beta = unname(b), coef_names = names(b), vcov = plain_matrix(stats::vcov(m)), logLik = as.numeric(logLik(m)), AIC = AIC(m), BIC = BIC(m), fitted = unname(fitted(m)), residuals = unname(residuals(m)), summary = paste(capture.output(summary(m)), collapse = '\n'))
  if (mixed) {
    ans$theta <- unname(lme4::getME(m, 'theta'))
    ans$VarCorr <- table_result(as.data.frame(lme4::VarCorr(m)))
    ans$singular <- lme4::isSingular(m)
    rr <- lme4::ranef(m, condVar = TRUE)
    ans$ranef <- lapply(rr, function(z) list(levels = rownames(z), names = names(z), values = plain_matrix(z), conditional_variance = attr(z, 'postVar')))
    ans$deviance <- if (s$call == 'lmer') -2 * as.numeric(logLik(m)) else deviance(m)
    ans$sigma <- sigma(m)
    ans$coefficients <- table_result(coef(summary(m)))
  } else if (inherits(m, 'gls')) {
    ans$sigma <- m$sigma
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
  ans <- tryCatch(withCallingHandlers({
    if ((s$operation %or% '') == 'versions') {
      ps <- c('jsonlite','lme4','lmerTest','car','emmeans','pbkrtest','nlme')
      list(R = R.version.string, packages = setNames(lapply(ps, function(p) as.character(packageVersion(p))), ps))
    } else {
      options(contrasts = if ((s$contrasts %or% 'treatment') == 'sum') c('contr.sum','contr.poly') else c('contr.treatment','contr.poly'))
      d <- make_data(s)
      if ((s$operation %or% '') == 'data') {
        list(columns = names(d), data = lapply(d, function(z) if (is.factor(z)) as.character(z) else z))
      } else {
        m <- fit_model(s, d)
        op <- s$operation %or% 'fit'
        if (op == 'fit') extract_model(m, s)
        else if (op == 'anova') table_result(stats::anova(m, type = s$type %or% 3L, ddf = s$ddf %or% 'Satterthwaite'))
        else if (op == 'Anova') {
          a <- list(mod = m, type = s$type %or% 2L)
          if (!is.null(s$test)) a$test.statistic <- s$test
          table_result(do.call(car::Anova, a))
        } else if (op %in% c('emmeans','emtrends','joint_tests')) {
          a <- s$emm_args %or% list()
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
          nd <- as.data.frame(lapply(s$newdata, unlist), stringsAsFactors = FALSE)
          if (inherits(m, 'merMod')) list(prediction = unname(predict(m, newdata = nd, re.form = if (isTRUE(s$population)) NA else NULL, allow.new.levels = s$allow_new_levels %or% FALSE, type = s$prediction_type %or% 'response')))
          else list(prediction = unname(predict(m, newdata = nd)))
        } else stop('Unapproved extraction operation')
      }
    }
  }, warning = function(w) { ws <<- c(ws, conditionMessage(w)); invokeRestart('muffleWarning') }, message = function(m) { ws <<- c(ws, conditionMessage(m)); invokeRestart('muffleMessage') }), error = function(e) list(error = conditionMessage(e)))
  list(id = id, result = ans, warnings = unique(ws))
}
cases <- input$cases %or% list(input)
output <- lapply(cases, run_one)
js <- toJSON(output, auto_unbox = TRUE, digits = 16, null = 'null', na = 'null', pretty = TRUE, matrix = 'rowmajor')
if (length(args) > 1L) writeLines(js, args[[2]]) else cat(js, '\n')
