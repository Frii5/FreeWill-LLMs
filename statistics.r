library(jsonlite)
library(PlackettLuce)

fit_folder <- function(input_folder, output_csv, items) {
  rankings_to_matrix <- function(rankings, items) {
    if (is.matrix(rankings) || is.data.frame(rankings)) {
      X <- matrix(0L, nrow = nrow(rankings), ncol = length(items))
      colnames(X) <- items

      for (r in seq_len(nrow(rankings))) {
        ranking <- as.character(rankings[r, ])
        ranking <- ranking[!is.na(ranking) & ranking != ""]
        X[r, ranking] <- seq_along(ranking)
      }
    } else {
      X <- matrix(0L, nrow = length(rankings), ncol = length(items))
      colnames(X) <- items

      for (r in seq_along(rankings)) {
        ranking <- as.character(unlist(rankings[[r]], use.names = FALSE))
        ranking <- ranking[!is.na(ranking) & ranking != ""]
        X[r, ranking] <- seq_along(ranking)
      }
    }

    X
  }

  fit_one_json <- function(path, items) {
    rankings <- fromJSON(path)

    X <- rankings_to_matrix(rankings, items)
    R <- as.rankings(X)
    mod <- PlackettLuce(R)
    worths <- coef(mod, log = FALSE)

    worths_full <- setNames(rep(NA_real_, length(items)), items)
    worths_full[names(worths)] <- worths

    data.frame(
      model = tools::file_path_sans_ext(basename(path)),
      t(as.data.frame(worths_full)),
      row.names = NULL,
      check.names = FALSE
    )
  }

  json_files <- list.files(input_folder, pattern = "\\.json$", full.names = TRUE)

  if (length(json_files) == 0) {
    stop("No JSON files found in: ", input_folder)
  }

  worth_rows <- lapply(json_files, fit_one_json, items = items)
  worth_table <- do.call(rbind, worth_rows)

  write.csv(worth_table, output_csv, row.names = FALSE)

  cat("Wrote:", output_csv, "\n")
  print(worth_table)

  invisible(worth_table)
}

# -----------------------------
# Part I
# -----------------------------
items_part1 <- c(
  "FW1","FW2","FW3","FW4","FW5",
  "DE1","DE2","DE3","DE4","DE5",
  "DU1","DU2","DU3","DU4","DU5"
)

fit_folder(
  input_folder = "rankings_json_part1",
  output_csv = "worths_by_model_part1.csv",
  items = items_part1
)

# -----------------------------
# Part II
# -----------------------------
items_part2 <- c(
  "FC1","FC2","FC3","FC4","FC5","FC6","FC7",
  "MC1","MC2","MC3","MC4","MC5","MC6","MC7"
)

fit_folder(
  input_folder = "rankings_json_part2",
  output_csv = "worths_by_model_part2.csv",
  items = items_part2
)