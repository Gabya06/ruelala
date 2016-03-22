# saw charts here
#http://analyzecore.com/2015/12/10/cohort-analysis-retention-rate-visualization-r/

# import data with counts by year, month, segment and actions
actions_grouped = read.csv("/Users/Gabi/dev/Shareablee/RueLaLa/data/yr_month_seg_actions.csv")
colnames(actions_grouped) = c('year','month','segment','actiontype', 'count')


# add  0 in front of months
actions_grouped <- actions_grouped %>% 
  mutate(month, month = ifelse(month < 10, paste(0, month, sep=''),month )) %>% 
  mutate(year_month = paste(year, month, sep = ""))

likes <- actions_grouped %>% filter( actiontype == 'like')
View(likes)
head(likes)

# plot likes
ggplot(na.omit(likes), aes(x = year, y = count, color = segment, size  = count)) +
  geom_point() + facet_wrap(~segment) + geom_smooth(method ='lm')


orders <- actions_grouped %>% filter( actiontype == 'Order')
head(orders)
# plot orders
ggplot(na.omit(orders), aes(x = year, y = count, color = segment, size  = count)) + 
  geom_point() + facet_wrap(~segment) + geom_smooth(method ='lm')



likes_plot <- likes %>% group_by(segment) %>% arrange(year_month) %>%
  mutate(number_prev_year = lag(count),
      count_201407 = count[which(year_month == '201407')]) %>%
  ungroup() %>%
  mutate(ret_rate_prev_year = count / number_prev_year,
         ret_rate = count / count_201407,
         year_seg = paste(year_month, segment, sep = '-'))


##### The second way for plotting cycle plot via multi-plotting
# plot #1 - Retention rate
# can change methose to loess
p1 <- ggplot(na.omit(likes_plot), aes(x = year_seg, y = ret_rate, group = year_month, color = year_month)) +
  theme_bw() +
  geom_point(size = 4) +
  geom_text(aes(label = percent(round(ret_rate, 2))),
            size = 4, hjust = 0.4, vjust = -0.6, face = "plain") +
  geom_smooth(size = 2.5, method = 'lm', color = 'darkred', aes(fill = year_month)) +
  theme(legend.position='none',
        plot.title = element_text(size=20, face="bold", vjust=2),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=18, face="bold"),
        axis.text = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(y = 'Retention Rate') +
  ggtitle("Customer Retention Rate - Cycle plot")

  
# plot #2 - number of customers
p2 <- ggplot(na.omit(likes_plot), aes(x = year_seg, group = year_month, color = year_month)) +
  theme_bw() +
  geom_bar(aes(y = number_prev_year, fill = year_month), alpha = 0.2, stat = 'identity') +
  geom_bar(aes(y = count, fill = year_month), alpha = 0.6, stat = 'identity') +
  geom_text(aes(y = number_prev_year, label = number_prev_year),
            angle = 90, size = 4, hjust = -0.1, vjust = 0.4) +
  geom_text(aes(y = count, label = count),
            angle = 90, size = 4, hjust = -0.1, vjust = 0.4) +
  geom_text(aes(y = 0, label = segment), color = 'white', angle = 90, size = 4, hjust = -0.05, vjust = 0.4) +
  theme(legend.position='none',
        plot.title = element_text(size=20, face="bold", vjust=2),
        axis.title.x = element_text(size=18, face="bold"),
        axis.title.y = element_text(size=18, face="bold"),
        axis.text = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_y_continuous(limits = c(0, max(likes_plot$count_201407 * 1.1))) +
  labs(x = 'Year of Lifetime by Cohorts', y = 'Number of Customers')

# multiplot
grid.arrange(p1, p2, ncol = 1)
# retention rate bubble chart
ggplot(na.omit(likes_plot), aes(x = segment, y = ret_rate, group = segment, color = year_month)) +
  theme_bw() +
  scale_size(range = c(15, 40)) +
  geom_line(size = 2, alpha = 0.3) +
  geom_point(aes(size = number_prev_year), alpha = 0.3) +
  geom_point(aes(size = count), alpha = 0.8) +
  geom_smooth(linetype = 2, size = 2, method = 'lm', aes(group = year_month, fill = year_month), alpha = 0.2) +
  geom_text(aes(label = paste0(count, '/', number_prev_year, '\n', percent(round(ret_rate, 2)))),
            color = 'white', size = 3, hjust = 0.5, vjust = 0.5, face = "plain") +
  theme(legend.position='none',
        plot.title = element_text(size=20, face="bold", vjust=2),
        axis.title.x = element_text(size=18, face="bold"),
        axis.title.y = element_text(size=18, face="bold"),
        axis.text = element_text(size=16),
        axis.text.x = element_text(size=10, angle=90, hjust=.5, vjust=.5, face="plain"),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(x = 'Cohorts', y = 'Retention Rate by Year of Lifetime') +
  ggtitle("Customer Retention Rate - Bubble chart")






orders_plot <- orders %>% group_by(segment) %>% arrange(year_month) %>%
  mutate(number_prev_year = lag(count),
         count_201407 = count[which(year_month == '201407')]) %>%
  ungroup() %>%
  mutate(ret_rate_prev_year = count / number_prev_year,
         ret_rate = count / count_201407,
         year_seg = paste(year_month, segment, sep = '-'))


p3 <- ggplot(na.omit(orders_plot), aes(x = year_seg, y = ret_rate, group = year_month, color = year_month)) +
  theme_bw() +
  geom_point(size = 4) +
  geom_text(aes(label = percent(round(ret_rate, 2))),
            size = 4, hjust = 0.4, vjust = -0.6, face = "plain") +
  geom_smooth(size = 1.5, method = 'loess', color = 'darkred', aes(fill = year_month)) +
  theme(legend.position='none',
        plot.title = element_text(size=20, face="bold", vjust=2),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=18, face="bold"),
        axis.text = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(y = 'Retention Rate') +
  ggtitle("Customer Retention Rate - Cycle plot")


# plot #2 - number of customers
p4 <- ggplot(na.omit(orders_plot), aes(x = year_seg, group = year_month, color = year_month)) +
  theme_bw() +
  geom_bar(aes(y = number_prev_year, fill = year_month), alpha = 0.2, stat = 'identity') +
  geom_bar(aes(y = count, fill = year_month), alpha = 0.6, stat = 'identity') +
  geom_text(aes(y = number_prev_year, label = number_prev_year),
            angle = 90, size = 4, hjust = -0.1, vjust = 0.4) +
  geom_text(aes(y = count, label = count),
            angle = 90, size = 4, hjust = -0.1, vjust = 0.4) +
  geom_text(aes(y = 0, label = segment), color = 'white', angle = 90, size = 4, hjust = -0.05, vjust = 0.4) +
  theme(legend.position='none',
        plot.title = element_text(size=20, face="bold", vjust=2),
        axis.title.x = element_text(size=18, face="bold"),
        axis.title.y = element_text(size=18, face="bold"),
        axis.text = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_y_continuous(limits = c(0, max(likes_plot$count_201407 * 1.1))) +
  labs(x = 'Year of Lifetime by Cohorts', y = 'Number of Customers')

# multiplot
grid.arrange(p3, p4, ncol = 1)