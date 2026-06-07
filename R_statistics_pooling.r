library(jsonlite)
library(PlackettLuce)
library(qvcalc)
library(ggplot2)
library(dplyr)

# Choose Part 1 or Part 2:
part <- "part2" 
setwd("C:/Users/Friis/OneDrive/UNI/Bachelor Projekt/FreeWill-LLMs")
if (part == "part1") {
  folder <- "rankings_json_part1"
  items <- c("FW1","FW2","FW3","FW4","FW5",
             "DE1","DE2","DE3","DE4","DE5",
             "DU1","DU2","DU3","DU4","DU5")
} else if (part == "part2") {
  folder <- "rankings_json_part2"
  items <- c("FC1","FC2","FC3","FC4","FC5","FC6","FC7",
             "MC1","MC2","MC3","MC4","MC5","MC6","MC7")
}

# Find JSON files:
json_files <- list.files(folder, pattern = "\\.json$", full.names = TRUE)
cat("Found", length(json_files), "JSON files for", part, "\n")

# Rankings into rank matrix for Plackett Luce:
rankings_to_matrix <- function(rankings_raw, items) {
  if (is.matrix(rankings_raw) || is.data.frame(rankings_raw)) {
    n <- nrow(rankings_raw)
  } else {
    n <- length(rankings_raw)
  }
  
  X <- matrix(0L, nrow = n, ncol = length(items))
  colnames(X) <- items
  
  for (r in seq_len(n)) {
    if (is.matrix(rankings_raw) || is.data.frame(rankings_raw)) {
      ranking <- as.character(rankings_raw[r, ])
    } else {
      ranking <- as.character(unlist(rankings_raw[[r]], use.names = FALSE))
    }
    ranking <- ranking[!is.na(ranking) & ranking != ""]
    
    if (length(ranking) < 2) next
    if (!all(ranking %in% items)) next
    
    X[r, ranking] <- seq_along(ranking)
  }
  X
}

# Pool rankings:
all_X_list <- lapply(json_files, function(f) fromJSON(f) |> rankings_to_matrix(items))
X_pooled <- do.call(rbind, all_X_list)
cat("Pooled rankings:", nrow(X_pooled), "\n")
print(data.frame(item = items, chosen = colSums(X_pooled > 0)), row.names = FALSE)

# Fit Plackett Luce:
R <- as.rankings(X_pooled)

prior <- list(
  mu = rep(0, ncol(X_pooled)),
  Sigma = diag(1, ncol(X_pooled))
)

mod <- PlackettLuce(R, normal = prior)
qv <- qvcalc(mod)
summary(qv)
plot(qv, ylab = "Worth (log)", main = NULL)

#############################################
# General Linear Hypothesis Tests for Results:
library(multcomp)

b <- coef(mod, log = TRUE)
V <- vcov(mod)

if (is.null(rownames(V)) || is.null(colnames(V))) {
  dimnames(V) <- list(names(b), names(b))
}

pair_glht <- function(item1, item2, mod, b = coef(mod, log = TRUE), V = vcov(mod)) {
  if (is.null(rownames(V)) || is.null(colnames(V))) {
    dimnames(V) <- list(names(b), names(b))
  }
  
  if (!item1 %in% names(b)) {
    stop(item1, " is not in coef(mod, log = TRUE).")
  }
  
  if (!item2 %in% names(b)) {
    stop(item2, " is not in coef(mod, log = TRUE).")
  }
  
  K <- matrix(
    0,
    nrow = 1,
    ncol = length(b),
    dimnames = list(paste(item1, "-", item2), names(b))
  )
  
  K[1, item1] <-  1
  K[1, item2] <- -1
  
  glht_res <- glht(
    mod,
    linfct = K,
    coef. = b,
    vcov. = V
  )
  
  summary(glht_res)
}

#Insert Pairs here:
pair_glht("MC6", "MC7", mod, b, V)
##################################

#Bootstrapping normalized worths:
B <- 1000
set.seed(123)

prior <- list(
  mu = rep(0, ncol(X_pooled)),
  Sigma = diag(1, ncol(X_pooled))
)

#Fitting on original data:
R <- as.rankings(X_pooled)
mod <- PlackettLuce(R, normal = prior)

est <- coef(mod, log = FALSE)
items <- names(est)

#Bootstrap
boot_worth <- matrix(
  NA_real_,
  nrow = B,
  ncol = length(est),
  dimnames = list(NULL, items)
)

n <- nrow(X_pooled)

for (b in seq_len(B)) {
  idx <- sample.int(n, size = n, replace = TRUE)
  Xb <- X_pooled[idx, , drop = FALSE]
  
  fit_b <- try({
    Rb <- as.rankings(Xb)
    mod_b <- PlackettLuce(Rb, normal = prior)
    coef(mod_b, log = FALSE)
  }, silent = TRUE)
  
  if (!inherits(fit_b, "try-error")) {
    boot_worth[b, names(fit_b)] <- fit_b
  }
}

boot_worth <- boot_worth[complete.cases(boot_worth), , drop = FALSE]

cat("Successful bootstrap fits:", nrow(boot_worth), "out of", B, "\n")

#Percentiles
ci <- t(apply(boot_worth, 2, quantile, probs = c(0.025, 0.975)))
ci <- ci[items, , drop = FALSE]

#Plot
xpos <- seq_along(items)

plot(
  xpos, est,
  xaxt = "n",
  xlab = "",
  ylab = "Worth",
  pch = 19,
  ylim = range(c(ci[, 1], ci[, 2]), na.rm = TRUE)
)

segments(
  x0 = xpos,
  y0 = ci[, 1],
  x1 = xpos,
  y1 = ci[, 2]
)

axis(1, at = xpos, labels = items, las = 2)

data.frame(worth = est, lower = ci[,1], upper = ci[,2])
#######################################################








