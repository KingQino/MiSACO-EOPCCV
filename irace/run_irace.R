library(irace)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
  setwd(dirname(script_path))
}

message("Starting irace from elite configurations...")
irace.cmdline("--scenario scenario.txt")

if (file.exists("irace.Rdata")) {
  load("irace.Rdata")

  best_id <- tail(iraceResults$iterationElites, 1)[[1]][1]
  best <- iraceResults$allConfigurations[best_id, ]
  write.table(best, file = "best-config.txt", sep = "\t", quote = FALSE, row.names = FALSE)

  write.csv(iraceResults$allConfigurations, file = "all-configs.csv", row.names = FALSE)

  last_gen_ids <- tail(iraceResults$iterationElites, 1)[[1]]
  last_gen <- iraceResults$allConfigurations[last_gen_ids, ]
  write.csv(last_gen, file = "last-generation.csv", row.names = FALSE)

  if (!is.null(iraceResults$experimentLog)) {
    write.csv(iraceResults$experimentLog, file = "experiment-log.csv", row.names = FALSE)
  }
}
