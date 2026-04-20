library(jsonlite)
library(PlackettLuce)
library(qvcalc)
library(ggplot2)
library(dplyr)

# =========================================================
# 1. Choose part
# =========================================================
part <- "part2"   # ← change to "part1" when needed

# Set folder and items
if (part == "part1") {
  folder <- "rankings_json_part1"
  items <- c("FW1","FW2","FW3","FW4","FW5","DE1","DE2","DE3","DE4","DE5","DU1","DU2","DU3","DU4","DU5")
} else if (part == "part2") {
  folder <- "rankings_json_part2"
  items <- c("FC1","FC2","FC3","FC4","FC5","FC6","FC7","MC1","MC2","MC3","MC4","MC5","MC6","MC7")
} else {
  stop("part must be 'part1' or 'part2'")
}

# =========================================================
# 2. Find JSON files
# =========================================================
json_files <- list.files(folder, pattern = "\\.json$", full.names = TRUE)
cat("Found", length(json_files), "JSON files for", part, "\n")

# =========================================================
# 3. Ranking matrix function
# =========================================================
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

# =========================================================
# 4. Pool data
# =========================================================
all_X_list <- lapply(json_files, function(f) fromJSON(f) |> rankings_to_matrix(items))
X_pooled <- do.call(rbind, all_X_list)

cat("Pooled rankings:", nrow(X_pooled), "\n")

# Frequency check
print(data.frame(item = items, chosen = colSums(X_pooled > 0)), row.names = FALSE)

# =========================================================
# 5. Fit model
# =========================================================
R <- as.rankings(X_pooled)

mod <- PlackettLuce(R)

summary(mod)

qv <- qvcalc(mod)
plot(qv, ylab = "Worth (log)", main = NULL)

coef(mod)
logLik(mod)


data.frame(
  item = colnames(X_pooled),
  appearances = colSums(X_pooled > 0),
  first_places = colSums(X_pooled == 1),
  last_places = sapply(seq_len(ncol(X_pooled)), function(j) {
    sum(X_pooled[, j] == apply(X_pooled, 1, max))
  })
)

prior <- list(
  mu = rep(0, ncol(X_pooled)),
  Sigma = diag(9, ncol(X_pooled))
)

mod <- PlackettLuce(R, normal = prior)
summary(mod)

worths <- coef(mod, log = FALSE)







library(PlackettLuce)

# -----------------------------
# settings
# -----------------------------
B <- 1000
set.seed(123)

prior <- list(
  mu = rep(0, ncol(X_pooled)),
  Sigma = diag(9, ncol(X_pooled))
)

# -----------------------------
# fit on original pooled data
# -----------------------------
R <- as.rankings(X_pooled)
mod <- PlackettLuce(R, normal = prior)

est <- coef(mod, log = FALSE)
items <- names(est)

# -----------------------------
# bootstrap
# -----------------------------
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

# -----------------------------
# percentile intervals
# -----------------------------
ci <- t(apply(boot_worth, 2, quantile, probs = c(0.025, 0.975)))
ci <- ci[items, , drop = FALSE]

# -----------------------------
# simple plot
# -----------------------------
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


##########################

library(jsonlite)
library(PlackettLuce)

part <- "part2"
model_name <- "grok-4.20-0309-reasoning"
B <- 500
set.seed(123)

if (part == "part1") {
  folder <- "rankings_json_part1"
  items <- c("FW1","FW2","FW3","FW4","FW5",
             "DE1","DE2","DE3","DE4","DE5",
             "DU1","DU2","DU3","DU4","DU5")
} else {
  folder <- "rankings_json_part2"
  items <- c("FC1","FC2","FC3","FC4","FC5","FC6","FC7",
             "MC1","MC2","MC3","MC4","MC5","MC6","MC7")
}

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
    X[r, ranking] <- seq_along(ranking)
  }
  
  X
}

json_file <- list.files(folder, pattern = model_name, full.names = TRUE)[1]
rankings_raw <- fromJSON(json_file)
X_model <- rankings_to_matrix(rankings_raw, items)

prior <- list(
  mu = rep(0, ncol(X_model)),
  Sigma = diag(9, ncol(X_model))
)

R <- as.rankings(X_model)
mod <- PlackettLuce(R, normal = prior)

est <- coef(mod, log = FALSE)

boot_worth <- matrix(NA_real_, nrow = B, ncol = length(est))
colnames(boot_worth) <- names(est)

n <- nrow(X_model)

for (b in seq_len(B)) {
  idx <- sample.int(n, n, replace = TRUE)
  Xb <- X_model[idx, , drop = FALSE]
  mod_b <- PlackettLuce(as.rankings(Xb), normal = prior)
  boot_worth[b, ] <- coef(mod_b, log = FALSE)
}

ci <- t(apply(boot_worth, 2, quantile, probs = c(0.025, 0.975)))

xpos <- seq_along(est)

plot(
  xpos, est,
  xaxt = "n",
  xlab = "",
  ylab = "Worth",
  pch = 19,
  ylim = range(c(ci[,1], ci[,2]))
)

arrows(
  x0 = xpos, y0 = ci[,1],
  x1 = xpos, y1 = ci[,2],
  angle = 90, code = 3, length = 0.05
)

axis(1, at = xpos, labels = names(est), las = 2)






