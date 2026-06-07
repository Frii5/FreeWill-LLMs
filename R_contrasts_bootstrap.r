library(jsonlite)
library(PlackettLuce)

# Same code as in the pooling script:
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

json_files <- list.files(folder, pattern = "\\.json$", full.names = TRUE)

cat("Found", length(json_files), "JSON files for", part, "\n")

rankings_to_matrix <- function(rankings_raw, items) {
  rankings <- if (is.matrix(rankings_raw) || is.data.frame(rankings_raw)) {
    split(as.data.frame(rankings_raw), seq_len(nrow(rankings_raw)))
  } else {
    rankings_raw
  }
  
  X <- matrix(0L, nrow = length(rankings), ncol = length(items),
              dimnames = list(NULL, items))
  
  for (r in seq_along(rankings)) {
    ranking <- as.character(unlist(rankings[[r]], use.names = FALSE))
    ranking <- ranking[!is.na(ranking) & ranking != ""]
    
    if (length(ranking) >= 2 && all(ranking %in% items)) {
      X[r, ranking] <- seq_along(ranking)
    }
  }
  
  X
}

all_X_list <- lapply(json_files, function(file) {
  rankings_raw <- fromJSON(file)
  rankings_to_matrix(rankings_raw, items)
})

model_names <- tools::file_path_sans_ext(basename(json_files))
names(all_X_list) <- model_names

X_pooled <- do.call(rbind, all_X_list)

###################################
# Bootstrapping, Worth-Contrasts
####################################

# Model names:
model_names <- tools::file_path_sans_ext(basename(json_files))
names(all_X_list) <- model_names

# Define each category and residual:
category_list <- list(
  incompatibilist = c("FC1", "FC4", "FC6", "MC1", "MC3"),
  compatibilist   = c("MC4", "MC5", "FC2", "FC3", "MC6", "FC7"),
  neutral         = c("FC5", "MC7", "MC2")
)

item_category <- setNames(rep(NA_character_, length(items)), items)

for (category in names(category_list)) {
  item_category[category_list[[category]]] <- category
}

category_table <- data.frame(
  item = names(item_category),
  category = item_category,
  row.names = NULL
)

print(category_table)

##################################
# Functions to fit resampled model:
B <- 1000
set.seed(123)

prior <- list(
  mu = rep(0, length(items)),
  Sigma = diag(1, length(items))
)

fit_worths <- function(X) {
  mod <- PlackettLuce(as.rankings(X), normal = prior)
  coef(mod, log = FALSE)
}

category_scores <- function(worths) {
  sums <- tapply(worths, item_category[names(worths)], sum)
  
  compat_sum <- sums["compatibilist"]
  incomp_sum <- sums["incompatibilist"]
  neutral_sum <- sums["neutral"]
  
  c(
    mass_contrast = unname(compat_sum - incomp_sum),
    
    mean_contrast = unname(
      compat_sum / length(category_list$compatibilist) -
        incomp_sum / length(category_list$incompatibilist)
    ),
    
    compatibilist_sum = unname(compat_sum),
    incompatibilist_sum = unname(incomp_sum),
    neutral_sum = unname(neutral_sum)
  )
}

summarise_boot <- function(x) {
  c(
    q025 = unname(quantile(x, 0.025, na.rm = TRUE)),
    median = unname(quantile(x, 0.500, na.rm = TRUE)),
    q975 = unname(quantile(x, 0.975, na.rm = TRUE)),
    mean = mean(x, na.rm = TRUE),
    prob_positive = mean(x > 0, na.rm = TRUE)
  )
}

###########################
# Original model fit scores:
original_scores <- do.call(rbind, lapply(names(all_X_list), function(model) {
  worths <- fit_worths(all_X_list[[model]])
  scores <- category_scores(worths)
  
  data.frame(
    model = model,
    t(scores),
    row.names = NULL
  )
}))

print(original_scores)

#######################
### BOOTSTRAP START ###

boot_results <- vector("list", B * length(all_X_list))
counter <- 1

for (b in seq_len(B)) {
  if (b %% 50 == 0) {
    cat("Bootstrap", b, "of", B, "\n")
  }
  
  for (model in names(all_X_list)) {
    X <- all_X_list[[model]]
    n <- nrow(X)
    
    idx <- sample.int(n, size = n, replace = TRUE)
    Xb <- X[idx, , drop = FALSE]
    
    fit_b <- try({
      worths_b <- fit_worths(Xb)
      scores_b <- category_scores(worths_b)
      
      data.frame(
        bootstrap = b,
        model = model,
        t(scores_b),
        row.names = NULL
      )
    }, silent = TRUE)
    
    if (!inherits(fit_b, "try-error")) {
      boot_results[[counter]] <- fit_b
      counter <- counter + 1
    }
  }
}

boot_results <- do.call(rbind, boot_results)

cat(
  "Successful bootstrap fits:",
  nrow(boot_results),
  "out of",
  B * length(all_X_list),
  "\n"
)

################
# SUMMARIES

per_model_mass_summary <- do.call(rbind, lapply(
  split(boot_results$mass_contrast, boot_results$model),
  summarise_boot
))

per_model_mass_summary <- data.frame(
  model = rownames(per_model_mass_summary),
  per_model_mass_summary,
  row.names = NULL
)

per_model_mean_summary <- do.call(rbind, lapply(
  split(boot_results$mean_contrast, boot_results$model),
  summarise_boot
))

per_model_mean_summary <- data.frame(
  model = rownames(per_model_mean_summary),
  per_model_mean_summary,
  row.names = NULL
)

cat("\nPer-model mass contrast summary:\n")
print(per_model_mass_summary)

cat("\nPer-model mean-item contrast summary:\n")
print(per_model_mean_summary)

##################
# Global summaries

global_total_mass <- tapply(
  boot_results$mass_contrast,
  boot_results$bootstrap,
  sum,
  na.rm = TRUE
)

global_mean_mass <- tapply(
  boot_results$mass_contrast,
  boot_results$bootstrap,
  mean,
  na.rm = TRUE
)

global_total_mean_item <- tapply(
  boot_results$mean_contrast,
  boot_results$bootstrap,
  sum,
  na.rm = TRUE
)

global_mean_mean_item <- tapply(
  boot_results$mean_contrast,
  boot_results$bootstrap,
  mean,
  na.rm = TRUE
)

cat("\nGlobal total mass contrast:\n")
print(summarise_boot(global_total_mass))

cat("\nGlobal mean mass contrast:\n")
print(summarise_boot(global_mean_mass))

cat("\nGlobal total mean-item contrast:\n")
print(summarise_boot(global_total_mean_item))

cat("\nGlobal mean mean-item contrast:\n")
print(summarise_boot(global_mean_mean_item))


# Correlation tests + Linear model

compat <- original_scores$mean_contrast

intel <- c(
  31, 46, 43, 32, 46,
  57, 6, 19, 23, 24,
  35, 39, 49, 6, 23,
  21, 24, 21, 10, 14
)

cor_data <- data.frame(
  model = original_scores$model,
  intelligence = intel,
  compatibilist_score = compat
)

print(cor_data)

cat("\nPearson correlation:\n")
print(cor.test(cor_data$intelligence, cor_data$compatibilist_score, method = "pearson"))

cat("\nSpearman correlation:\n")
print(cor.test(cor_data$intelligence, cor_data$compatibilist_score, method = "spearman", exact = FALSE))

cat("\nMean compatibilist score:\n")
print(mean(cor_data$compatibilist_score))

entropy <- c(
  3.204, 2.829, 3.079, 2.88, 2.906,
  2.943, 3.095, 3.045, 3.045, 3.207,
  2.859, 3.006, 2.924, 2.93, 3.29,
  2.64, 2.973, 3.188, 2.904, 3.682
)

lm(compat ~ intel + entropy)
summary(ml)

# Save bootstrap results. These are used for the ridge plot
# See Plotting/ridgeplot.py

library(dplyr)
library(tidyr)

boot_to_long <- function(mat, metric_name) {
  as.data.frame(mat) %>%
    mutate(rep = row_number()) %>%
    pivot_longer(
      cols = -rep,
      names_to = "model",
      values_to = "value"
    ) %>%
    mutate(metric = metric_name)
}

ridge_boot <- bind_rows(
  boot_to_long(boot_mass, "Mass contrast"),
  boot_to_long(boot_mean, "Mean-item contrast"),
  
  data.frame(
    rep = seq_len(nrow(boot_mass)),
    model = "OVERALL",
    value = rowMeans(boot_mass),
    metric = "Mass contrast"
  ),
  
  data.frame(
    rep = seq_len(nrow(boot_mean)),
    model = "OVERALL",
    value = rowMeans(boot_mean),
    metric = "Mean-item contrast"
  )
)

ridge_points <- original_scores %>%
  transmute(
    model,
    `Mass contrast` = mass_contrast,
    `Mean-item contrast` = mean_contrast
  ) %>%
  pivot_longer(
    cols = -model,
    names_to = "metric",
    values_to = "estimate"
  ) %>%
  bind_rows(
    original_scores %>%
      summarise(
        `Mass contrast` = mean(mass_contrast),
        `Mean-item contrast` = mean(mean_contrast)
      ) %>%
      mutate(model = "OVERALL") %>%
      pivot_longer(
        cols = -model,
        names_to = "metric",
        values_to = "estimate"
      )
  )

write.csv(ridge_boot, "ridge_boot_part2.csv", row.names = FALSE)
write.csv(ridge_points, "ridge_points_part2.csv", row.names = FALSE)























