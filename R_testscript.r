library(jsonlite)
library(PlackettLuce)
library(qvcalc)

# =========================================================
# 1. Choose which part and one JSON file
# =========================================================
part <- "part2"
json_path <- "rankings_json_part2/gpt-5.4-mini.json"

# =========================================================
# 2. Set item universe by part
# =========================================================
if (part == "part1") {
  items <- c(
    "FW1","FW2","FW3","FW4","FW5",
    "DE1","DE2","DE3","DE4","DE5",
    "DU1","DU2","DU3","DU4","DU5"
  )
} else if (part == "part2") {
  items <- c(
    "FC1","FC2","FC3","FC4","FC5","FC6","FC7",
    "MC1","MC2","MC3","MC4","MC5","MC6","MC7"
  )
} else {
  stop("part must be 'part1' or 'part2'")
}

# =========================================================
# 3. Load the JSON
# =========================================================
rankings_raw <- fromJSON(json_path)

cat("\n=============================\n")
cat("RAW JSON CONTENT\n")
cat("=============================\n")
cat("Part:\n")
print(part)

cat("\nClass of rankings_raw object:\n")
print(class(rankings_raw))

cat("\nStructure of rankings_raw:\n")
str(rankings_raw)

cat("\nFirst few rankings:\n")
print(head(rankings_raw, 5))

# =========================================================
# 4. Convert rankings into ranking matrix
#    Each row = one observed ranking
#    Each column = one possible item
#    Ranked items get 1,2,...,K ; absent items get 0
# =========================================================
rankings_to_matrix <- function(rankings_raw, items) {
  if (is.matrix(rankings_raw) || is.data.frame(rankings_raw)) {
    X <- matrix(0L, nrow = nrow(rankings_raw), ncol = length(items))
    colnames(X) <- items

    for (r in seq_len(nrow(rankings_raw))) {
      ranking <- as.character(rankings_raw[r, ])
      ranking <- ranking[!is.na(ranking) & ranking != ""]

      if (length(ranking) < 2) {
        stop("Row ", r, " has fewer than 2 ranked items.")
      }

      if (!all(ranking %in% items)) {
        stop(
          "Unknown item(s) in row ", r, ": ",
          paste(ranking[!ranking %in% items], collapse = ", ")
        )
      }

      X[r, ranking] <- seq_along(ranking)
    }
  } else {
    X <- matrix(0L, nrow = length(rankings_raw), ncol = length(items))
    colnames(X) <- items

    for (r in seq_along(rankings_raw)) {
      ranking <- as.character(unlist(rankings_raw[[r]], use.names = FALSE))
      ranking <- ranking[!is.na(ranking) & ranking != ""]

      if (length(ranking) < 2) {
        stop("Row ", r, " has fewer than 2 ranked items.")
      }

      if (!all(ranking %in% items)) {
        stop(
          "Unknown item(s) in row ", r, ": ",
          paste(ranking[!ranking %in% items], collapse = ", ")
        )
      }

      X[r, ranking] <- seq_along(ranking)
    }
  }

  X
}

X <- rankings_to_matrix(rankings_raw, items)

cat("\n=============================\n")
cat("RANKING MATRIX X\n")
cat("=============================\n")
cat("Dimensions of X:\n")
print(dim(X))

cat("\nFirst 5 rows of X:\n")
print(X[1:min(5, nrow(X)), , drop = FALSE])

first_nonzero <- which(rowSums(X > 0) > 0)[1]
ranking_size <- sum(X[first_nonzero, ] > 0)

cat("\nDetected ranking size:\n")
print(ranking_size)

# =========================================================
# 5. Convert matrix to rankings object
# =========================================================
R <- as.rankings(X)

cat("\n=============================\n")
cat("RANKINGS OBJECT\n")
cat("=============================\n")
print(R)
cat("\nClass of R:\n")
print(class(R))

# =========================================================
# 6. Fit Plackett-Luce model
# =========================================================
mod <- PlackettLuce(R)

cat("\n=============================\n")
cat("MODEL OBJECT\n")
cat("=============================\n")
print(mod)

# =========================================================
# 7. Basic summary
# =========================================================
cat("\n=============================\n")
cat("MODEL SUMMARY\n")
cat("=============================\n")
mod_sum <- summary(mod)
print(mod_sum)

# =========================================================
# 8. Extract coefficients
# =========================================================
cat("\n=============================\n")
cat("COEFFICIENTS\n")
cat("=============================\n")

log_coef <- coef(mod)
cat("\nLog-worth coefficients:\n")
print(log_coef)

worth_coef <- coef(mod, log = FALSE)
cat("\nNormalized worth parameters:\n")
print(worth_coef)

cat("\nCheck that worths sum to 1:\n")
print(sum(worth_coef))

# =========================================================
# 9. Standard errors and Wald intervals
# =========================================================
V <- vcov(mod)
se <- sqrt(diag(V))

cat("\n=============================\n")
cat("STANDARD ERRORS\n")
cat("=============================\n")
print(se)

ci_log <- cbind(
  lower = log_coef - 1.96 * se,
  estimate = log_coef,
  upper = log_coef + 1.96 * se
)

cat("\n=============================\n")
cat("95% WALD CONFIDENCE INTERVALS\n")
cat("=============================\n")
print(ci_log)

# =========================================================
# 10. Quasi-variance output
# =========================================================
cat("\n=============================\n")
cat("QUASI-VARIANCE OUTPUT\n")
cat("=============================\n")

qv <- qvcalc(mod)
print(summary(qv))
print(qv$qvframe)

# =========================================================
# 11. Clean results table
# =========================================================
cat("\n=============================\n")
cat("CLEAN RESULTS TABLE\n")
cat("=============================\n")

results_table <- data.frame(
  item = names(log_coef),
  log_estimate = as.numeric(log_coef),
  SE = as.numeric(se),
  CI_lower_log = as.numeric(ci_log[, "lower"]),
  CI_upper_log = as.numeric(ci_log[, "upper"]),
  worth = as.numeric(worth_coef[names(log_coef)]),
  row.names = NULL
)

qv_frame <- qv$qvframe
qv_frame$item <- rownames(qv_frame)
rownames(qv_frame) <- NULL

results_table <- merge(results_table, qv_frame, by = "item", all.x = TRUE, sort = FALSE)

print(results_table)

cat("\nDone.\n")