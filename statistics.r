library(jsonlite)
library(PlackettLuce)

# ---- CONFIG ----
input_folder <- "rankings_json"
output_csv <- "worths_by_model.csv"

items <- c(
  "FW1","FW2","FW3","FW4","FW5",
  "DE1","DE2","DE3","DE4","DE5",
  "DU1","DU2","DU3","DU4","DU5"
)

# ---- HELPERS ----
triads_to_matrix <- function(triads, items) {
  X <- matrix(0L, nrow = length(triads), ncol = length(items))
  colnames(X) <- items

  for (r in seq_along(triads)) {
    triad <- triads[[r]]
    X[r, triad] <- seq_along(triad)   # 1,2,3 = best to worst
  }

  X
}

aggregate_triads <- function(triads) {
  keys <- vapply(triads, paste, collapse = ">", FUN.VALUE = character(1))
  tab <- table(keys)
  unique_triads <- strsplit(names(tab), ">", fixed = TRUE)

  list(
    triads = unique_triads,
    weights = as.numeric(tab)
  )
}

fit_one_json <- function(path, items) {
  triads <- fromJSON(path, simplifyVector = FALSE)

  agg <- aggregate_triads(triads)
  X <- triads_to_matrix(agg$triads, items)
  R <- as.rankings(X)

  mod <- PlackettLuce(R, weights = agg$weights)

  worths <- coef(mod, log = FALSE)

  # Ensure all items appear in fixed order
  worths_full <- setNames(rep(NA_real_, length(items)), items)
  worths_full[names(worths)] <- worths

  data.frame(
    model = tools::file_path_sans_ext(basename(path)),
    t(as.data.frame(worths_full)),
    row.names = NULL,
    check.names = FALSE
  )
}

# ---- MAIN ----
json_files <- list.files(input_folder, pattern = "\\.json$", full.names = TRUE)

if (length(json_files) == 0) {
  stop("No JSON files found in: ", input_folder)
}

worth_rows <- lapply(json_files, fit_one_json, items = items)
worth_table <- do.call(rbind, worth_rows)

write.csv(worth_table, output_csv, row.names = FALSE)

cat("Wrote:", output_csv, "\n")
print(worth_table)