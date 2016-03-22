library(igraph)
library(reshape2)
library(ggplot2)
library(ggthemes)
library(R.utils)

setwd("~/dev/Shareablee/RueLaLa/")
transition.matrix = read.csv("~/dev/Shareablee/RueLaLa/cleanData/transition_matrix.csv")
transition.matrix$X <- NULL
#colnames(transition.matrix)  <- c(1,2,3,4)
colnames(transition.matrix) <- c("like","comment","login","order")
row.names(transition.matrix) <- c("like","comment","login","order")

m_tran <- melt(transition.matrix, value.name = 'weight')
from_node <- rep(x = c("like","comment","login","order"), times = 4)
#from_node <- rep(x = c(1,2,3,4), times = 4)
to_node <- m_tran$variable

temp_matrix <- data.frame(from_node, stringsAsFactors = F)
temp_matrix$to_node <-to_node
temp_matrix$to_node <- sapply(temp_matrix$to_node, as.character)
temp_matrix$weight <- round(m_tran$weight,3)

g <- graph.data.frame(temp_matrix, directed = T)

# delete edges with weight = 1
g1<- delete.edges(graph = g, edges = which(E(g)$weight ==0))

par(mai=c(1,1,1,1))

par(mai=c(0.25,0.25,0.25,.5))
E(g1)$color<-ifelse(E(g1)$weight>.5, "red", "darkblue")
plot(g1,layout=layout.fruchterman.reingold, 
     vertex.color ="skyblue", vertex.size=60, vertex.label.color = 'darkblue',
     vertex.label= V(g1)$name,  vertex.label.cex = 1.1, vertex.frame.color='blue',
     edge.arrow.size=.45, edge.width = .35, edge.label = percent(E(g1)$weight),
     edge.curved = TRUE, edge.label.cex=.5) 


png(filename="segmentC_states_probs2.png", height=800, width=600)


plot(g,layout=layout.fruchterman.reingold,edge.width=E(g)$weight*5, edge.color = 'black', 
     vertex.color =rainbow(vcount(g)), vertex.size=70, 
     vertex.label=c('like','comment','login','order'), vertex.label.cex=1, edge.arrow.size=.5 )


# all action counts
action_counts = read.csv("~/dev/Shareablee/RueLaLa/data/action_counts.csv")
# correlations
corr_mat = cor(action_counts[c('Login','Order','comment','like')])
colnames(corr_mat) = capitalize(colnames(corr_mat))

corr_melt = melt(corr_mat, value.name = 'Correlation')
corr_melt$Var1 = capitalize(corr_melt$Var1)
corr_melt$Var2 = capitalize(corr_melt$Var2)

ggplot(data = corr_melt, aes(x = Var1, y = Var2, fill = Correlation)) + 
  geom_tile() + 
  theme(panel.background = element_rect(fill = "white")) +
  scale_fill_gradient2(low = 'lightcyan', mid = 'lightblue', 
                       high = 'midnightblue', midpoint = mu, name = 'Correlations', guide = 'colorbar',
                       limits = c(0, 0.89) ) + 
  labs(x = NULL, y = NULL) + 
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_tufte(base_family = "Verdana", base_size = 11)   


# the same thing but only including users w at least one fb engagement
# all action counts
action_counts_fb = read.csv("~/dev/Shareablee/RueLaLa/data/action_counts_fb.csv")
# correlations
corr_mat_fb = cor(action_counts_fb[c('Login','Order','comment','like')])
colnames(corr_mat_fb) = capitalize(colnames(corr_mat_fb))

corr_melt_fb = melt(corr_mat_fb, value.name = 'Correlation')
corr_melt_fb$Var1 = capitalize(corr_melt_fb$Var1)
corr_melt_fb$Var2 = capitalize(corr_melt_fb$Var2)

ggplot(data = corr_melt_fb, aes(x = Var1, y = Var2, fill = Correlation)) + 
  geom_tile() + 
  theme(panel.background = element_rect(fill = "white")) +
  scale_fill_gradient2(low = 'lightcyan', mid = 'lightblue', 
                       high = 'midnightblue', midpoint = mu, name = 'Correlations', guide = 'colorbar',
                       limits = c(0, 0.89) ) + 
  labs(x = NULL, y = NULL) + 
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  theme_tufte(base_family = "Verdana", base_size = 11)   