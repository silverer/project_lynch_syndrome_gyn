if (!require("pacman")) install.packages("pacman"); library(pacman)
p_load(dplyr, readxl)
thresh.vars <- c("risk ac death oc surg", "risk oc tubal ligation",
                 "U HSBO", "U hysterectomy", 'lifetime risk')

process.thresh.outputs <- function(thresh.outs){
  wtp = 100000
  strat_recode = read.csv("../data/strategy_info.csv")
  strats = as.vector(strat_recode$pretty_label)
  names(strats) = strat_recode$strat_label
  thresh.clean = thresh.outs %>% 
    mutate(pretty_strategy = recode(strategy, !!!strats)) %>% 
    dplyr::filter(!is.na(icers)) %>% 
    dplyr::filter(icers < wtp)
  
  if(thresh.clean$changed.param[1] == 'ec lifetime risk'){
    thresh.clean$param.value <- thresh.clean$lifetime.ec.risk
    
  }else if(thresh.clean$changed.param[1] == 'oc lifetime risk'){
    thresh.clean$param.value <- thresh.clean$lifetime.oc.risk
  }
  range.by.gene <- thresh.clean %>% 
    group_by(gene) %>% 
    summarise(min.gene = min(param.value),
              max.gene = max(param.value))
  min.list <- as.vector(range.by.gene$min.gene)
  names(min.list) <- range.by.gene$gene
  max.list <- as.vector(range.by.gene$max.gene)
  names(max.list) <- range.by.gene$gene
  
  #Select optimal strategy for each gene and parameter value
  optimal.only = thresh.clean %>% 
    dplyr::group_by(gene, param.value) %>% 
    dplyr::filter(row_number()==1) %>% 
    dplyr::select(c(gene, pretty_strategy, param.value, changed.param, icers))
  
  optimal.only = ungroup(optimal.only)
  optimal.only = optimal.only %>% 
    mutate(min_param_val = recode(gene, !!!min.list),
           max_param_val = recode(gene, !!!max.list))
  return(optimal.only)
}
all.thresh.outs <- data.frame()
for(t in thresh.vars){
  tmp <- read.csv(paste0("../model_outs/threshold_icers_",
                         t, "_all_genes_05_20_21.csv"))
  if(t == 'lifetime risk'){
    tmp1 <- tmp %>% 
      dplyr::filter(changed.param == 'ec lifetime risk')
    tmp2 <- tmp %>% 
      dplyr::filter(changed.param == 'oc lifetime risk')
    cleaned1 <- process.thresh.outputs(tmp1)
    cleaned2 <- process.thresh.outputs(tmp2)
    cleaned <- rbind(cleaned1, cleaned2)
  }else{
    cleaned <- process.thresh.outputs(tmp)
  }
  
  all.thresh.outs <- rbind(all.thresh.outs, cleaned)
}

ranges.only <- all.thresh.outs %>% 
  group_by(gene, changed.param) %>% 
  dplyr::filter(!duplicated(pretty_strategy))

range.by.gene.strat <- all.thresh.outs %>% 
  group_by(gene, changed.param, pretty_strategy) %>% 
  summarise(min.gene.strat = min(param.value),
            max.gene.strat = max(param.value))

range.by.gene.strat <- ungroup(range.by.gene.strat)
range.by.gene.strat <- range.by.gene.strat %>% 
  group_by(gene,changed.param) %>% 
  arrange(min.gene.strat)

ranges.merged <- left_join(ranges.only, range.by.gene.strat, 
                           by = c("gene", "changed.param", 'pretty_strategy'))
prettify_range <- function(minval, maxval){
  return(paste(round(minval, 2), "-", round(maxval, 2)))
}
ranges.merged["strat_range"] = mapply(prettify_range, ranges.merged$min.gene.strat, 
                                      ranges.merged$max.gene.strat)

write.csv(ranges.merged, '../model_outs/threshold_icers_processed_ranges_R_05_20_21.csv')
