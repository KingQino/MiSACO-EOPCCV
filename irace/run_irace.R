library(irace)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
  setwd(dirname(script_path))
}

result_file <- readScenario("scenario.txt")$logFile
result_tag <- tools::file_path_sans_ext(basename(result_file))

message("Starting all-instance FE=600 irace from elite configurations...")
irace.cmdline("--scenario scenario.txt")

if (file.exists(result_file)) {
  load(result_file)

  best_id <- tail(iraceResults$iterationElites, 1)[[1]][1]
  best <- iraceResults$allConfigurations[best_id, ]
  write.table(best, file = paste0("best-config-", result_tag, ".txt"),
              sep = "\t", quote = FALSE, row.names = FALSE)

  write.csv(iraceResults$allConfigurations,
            file = paste0("all-configs-", result_tag, ".csv"), row.names = FALSE)

  last_gen_ids <- tail(iraceResults$iterationElites, 1)[[1]]
  last_gen <- iraceResults$allConfigurations[last_gen_ids, ]
  write.csv(last_gen, file = paste0("last-generation-", result_tag, ".csv"),
            row.names = FALSE)

  if (!is.null(iraceResults$experimentLog)) {
    write.csv(iraceResults$experimentLog,
              file = paste0("experiment-log-", result_tag, ".csv"),
              row.names = FALSE)
  }
}
