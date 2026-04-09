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

if (part == "part2") {
  prior <- list(mu = rep(0, length(items)), Sigma = diag(rep(9, length(items))))
  mod <- PlackettLuce(R, normal = prior, maxit = 2000)
  cat("\nFitted with normal prior (variance = 9) for Part 2\n")
} else {
  mod <- PlackettLuce(R, npseudo = 1, maxit = 1000)
}

summary(mod)

# =========================================================
# 6. Extract worths — FIXED item name handling
# =========================================================
worth_df <- tryCatch({
  qv <- qvcalc(mod)
  df <- as.data.frame(qv$qvframe) %>%
    tibble::rownames_to_column("item") %>%
    mutate(item = factor(item, levels = items))
}, error = function(e) {
  cat("qvcalc failed, using coef()\n")
  data.frame(item = items, estimate = coef(mod))
})

# Add worth and CI (safe version)
worth_df <- worth_df %>%
  mutate(
    worth = exp(estimate),
    lower = if ("quasiSE" %in% names(.)) exp(estimate - 1.96 * quasiSE) else NA_real_,
    upper = if ("quasiSE" %in% names(.)) exp(estimate + 1.96 * quasiSE) else NA_real_
  ) %>%
  arrange(desc(worth)) %>%
  mutate(item = factor(item, levels = item))   # lock sorted order

print(worth_df %>% select(item, worth, lower, upper), digits = 4)

# =========================================================
# 7. Clean Plot
# =========================================================
ggplot(worth_df, aes(x = item, y = worth)) +
  geom_point(size = 3.5, color = "steelblue") +
  {if (!all(is.na(worth_df$lower))) 
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.4, color = "steelblue")} +
  coord_flip() +
  scale_y_continuous(limits = c(0, NA)) +
  labs(title = paste("Pooled Plackett-Luce Item Worths —", part),
       subtitle = paste("Based on", nrow(X_pooled), "rankings from", length(json_files), "models"),
       y = "Worth (exp(estimate))",
       x = NULL,
       caption = if(all(is.na(worth_df$lower))) "Point estimates only" else "With 95% quasi-confidence intervals") +
  theme_minimal(base_size = 12) +
  theme(axis.text.y = element_text(size = 11))

ggsave(paste0("pooled_worths_", part, "_clean.png"), width = 11, height = 9, dpi = 300)
write.csv(worth_df, paste0("pooled_worths_", part, ".csv"), row.names = FALSE)