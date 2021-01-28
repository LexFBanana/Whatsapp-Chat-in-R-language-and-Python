
library(rwhatsapp)
library(lubridate)
library(tidyverse)
library(tidytext)
library(kableExtra)
library(RColorBrewer)
library(knitr)
library(qdapRegex)
## LIBS TO EXPORT AS HTML

library(plotly);
library(htmlwidgets);
# LIBRARY FOR EMOJI PNG IMAGE FETCH FROM https://abs.twimg.com
library(ggimage)

#LIBRARY FOR WORD

library(stopwords)
library(rvest)
library(ggplot2)


miChat <- rwa_read("C:\\DS File\\R\\bs.txt")
wdata <- rwa_read("C:\\DS File\\R\\bs.txt")



# Filter rows of <Media omitted> and messages that dont have authors
wdata <- wdata %>% filter(author != "" & text != "<Media omitted>" & !is.na(author))

#NUMBER OF MESSAGES [ MOST ACTIVE MEMBER ]
ggplotly(
  wdata %>%
    count(author) %>%
    ggplot(aes(x = reorder(author, n), y = n ,fill = author)) +
    geom_bar(stat = "identity") +
    ylab("Totals") + xlab("Group Members") +
    coord_flip() +
    ggtitle("Number of messages sent") +
    theme_minimal()
)




#DAYS OF WEEK MOST ACTIVE
wdata <- wdata %>% mutate(Dow =  wday(as.Date(wdata$time), label=TRUE))

dow <- wdata %>% filter(Dow !='') %>% group_by(Dow) %>% summarise(count = n())
ggplot(dow,aes(x=Dow,y = count, fill = Dow))+
  geom_bar(stat = "identity")+
  xlab("Days of the week")+
  ylab("Messages")+
  coord_flip()+
  geom_text(aes(label = scales::comma(count)), hjust = 3) +
  ggtitle("Days most active")+
  theme_minimal()





#MONTHS MOST ACTIVE AND TOP MEMBER
wdata <- wdata %>% mutate(months = month(as.POSIXct(wdata$time,'%m'),label = TRUE))

mnths <- wdata %>% filter(months !='') %>% group_by(months) %>% summarise(mcount = n())
actMember <- wdata %>% filter(months != '')%>% group_by(months,author)%>%summarise(scount = n())%>% slice(which.max(scount))
mnthsactMember <-  merge(mnths, actMember,by="months")


ggplot(mnthsactMember)+
  geom_bar(aes(x=months,y = mcount, fill = months),stat = "identity",width = 1)+
  geom_point(aes(x=months,y = scount,color = author),
             size = 4, alpha = 0.5,
             stat = "identity",
  )+
  # geom_text(aes(x=months,y = scount,label = Name), vjust = 0.5,hjust = -1,color ="white")+
  geom_label(aes(x=months,y = scount,label = paste0(author," (",scount,")")),
             fill = 'black', vjust = 0.5,hjust = -0.4,color ="white",alpha = 0.5,size = 3.5
  )+
  xlab("Months")+
  ylab("Messages")+
  coord_flip()+
  ggtitle("MONTH ACTIVITY AND MOST ACTIVE MEMBER EACH MONTH")+
  theme_minimal(base_size = 10)



# LINKS SHARED
links <- wdata %>%  select(text) %>% transmute(url = rm_url(wdata$text,extract = TRUE)) %>% filter(url != "") 
links$url <- as.character(links$url)
links$url <- urltools::domain(links$url)
links <- links %>% group_by(url) %>% summarise(count = n())
linksTop10 <- links  %>%  top_n(10) %>% arrange(desc(count))
ggplotly(
    linksTop10 %>%
        count(url,count) %>%
        ggplot(aes(x = reorder(url, count), y = count ,fill = url)) +
        geom_bar(stat = "identity") +
        ylab("Totals") + xlab("Link Shared") +
        coord_flip() +
        ggtitle("Number of link sent") +
        theme_minimal()
)




# EMOJI RANKING
plotEmojis <- miChat %>% 
  unnest(emoji, emoji_name) %>% 
  mutate( emoji = str_sub(emoji, end = 1)) %>% # REMOVE LINKS
  mutate( emoji_name = str_remove(emoji_name, ":.*")) %>% # REMOVE LIGATURE NAMES
  count(emoji, emoji_name) %>% 
  # PLOT TOP 20 EMOJIS
  top_n(30, n) %>% 
  arrange(desc(n)) %>% 
  # CREATE AN IMAGE URL WITH THE EMOJI UNICODE
  mutate( emoji_url = map_chr(emoji, 
                              ~paste0( "https://abs.twimg.com/emoji/v2/72x72/", as.hexmode(utf8ToInt(.x)),".png")) 
  )
# PLOT OF THE MOST USED EMOJIS RANKING
plotEmojis %>% 
  ggplot(aes(x=reorder(emoji_name, n), y=n)) +
  geom_col(aes(fill=n), show.legend = FALSE, width = .2) +
  geom_point(aes(color=n), show.legend = FALSE, size = 3) +
  geom_image(aes(image=emoji_url), size=.030) +
  scale_fill_gradient(low="#2b83ba",high="#d7191c") +
  scale_color_gradient(low="#2b83ba",high="#d7191c") +
  ylab("Number of times the emoji was used") +
  xlab("Emoji and meaning") +
  ggtitle("Most commonly used emojis", "Emojis most used by everyone") +
  coord_flip() +
  theme_minimal() +
  theme()

ggplotly()




# used emojis per user
plotEmojis <- miChat %>%
  unnest(emoji, emoji_name) %>%
  mutate( emoji = str_sub(emoji, end = 1)) %>% # 
  count(author, emoji, emoji_name, sort = TRUE) %>%
  # PLOT OF THE TOP 8 EMOJIS PER USER
  group_by(author) %>%
  top_n(n = 8, n) %>%
  slice(1:8) %>% 
  # CREATE AN IMAGE URL WITH THE EMOJI UNICODE
  mutate( emoji_url = map_chr(emoji, 
                              ~paste0("https://abs.twimg.com/emoji/v2/72x72/",as.hexmode(utf8ToInt(.x)),".png")) )

 # DATA PLOT
plotEmojis %>% 
  ggplot(aes(x = reorder(emoji, -n), y = n)) +
  geom_col(aes(fill = author, group=author), show.legend = FALSE, width = .15) +
  # USE TO FETCH AN EMOJI PNG IMAGE https://abs.twimg.com
  geom_image(aes(image=emoji_url), size=0.10) +
  ylab("Number of times the emoji was used") +
  xlab("Emoji") +
  facet_wrap(~author, ncol = 8, scales = "free")  +
  ggtitle("Most used emojis in conversation, by user") +
  theme_minimal() +
  theme(axis.text.x = element_blank())

ggplotly()





 #WE REMOVE WORDS WITHOUT RELEVANT MEANING, LIKE ARTICLES, PRONOUNS, ETC.
remover_palabras <- c(stopwords(language = "pt"),
                      "multimedia "," and "," the "," the "," in "," is "," yes "," it "," already "," but "," that ",
                      "the", "I", "my", "a", "with", "the", "omitted", "more", "that", "al", "an",
                      "del", "what", "all", "thus", "him", "his", "goes", "because", "all", "there are", "them",
                      "pue", "that", "are", "is", "well", "there", "yes", "see", "you are", "something", "you go",
                      "go", "I'm going", "I think", "was", "only", "nor", "only", "nothing", "here", "q", "you", "fez")


# WORDS COUNT
miChat %>%
  unnest_tokens(input = text, output = word) %>%
  filter(!word %in% remover_palabras) %>% 
  count(word) %>% 
  # PLOT OF THE TOP 20 MOST USED WORDS IN CONVERSATION
  top_n(30,n) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(x=reorder(word,n), y=n, fill=n, color=n)) +
  geom_col(show.legend = FALSE, width = .1) +
  geom_point(show.legend = FALSE, size = 3) +
  scale_fill_gradient(low="#2b83ba",high="#d7191c") +
  scale_color_gradient(low="#2b83ba",high="#d7191c") +
  ggtitle("Words most used in conversation in general") +
  xlab("Words") +
  ylab("Number of times the word was used") +
  coord_flip() +
  theme_minimal()
  ggplotly() 


  
  
  
# HTML PAGE FETCH EMOJI SENTIMENT RANKING 1.0
url_base <- "http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html"
doc <- read_html(url_base)
# SEARCH EMOJI AND PROCESS TABLE
tabla_emojis <- doc %>% 
  html_node("#myTable") %>% 
  html_table() %>% 
  as_tibble()
# UN PRIOR LOOK TO THE RESULT
tabla_emojis %>% 
  head(5) %>% 
  kable() %>% 
  kable_styling(font_size = 10)
# GET FEELING SCORE AND CLEAR NAMES FROM EMOJI TABLES
sentimiento_emoji <- tabla_emojis %>% 
  select(1,6:9) %>% 
  set_names("char", "negative","neutral","positive","sent.score")
# EXTRACT EMOJI AND UNITE WITH FEELING
emoji_chat <- miChat %>% 
  unnest(emoji, emoji_name) %>% 
  mutate( emoji = str_sub(emoji, end = 1)) %>% # Remove ligatures
  inner_join(sentimiento_emoji, by=c("emoji"="char")) 
# DISPLAY
emoji_chat %>% 
  select(-source, -day, -estacion) %>% 
  slice(1207:1219) %>% 
  head(10) %>%
  kable() %>% 
  kable_styling(font_size = 10)
# OCCURRENCES OF FEELINGS BY EMOJIS, PER USER
emoji_feeling_user <- emoji_chat %>% 
  group_by(author) %>% 
  summarise(
    positive=mean(positive),
    negative=mean(negative),
    neutral=mean(neutral),
    balance=mean(sent.score)
  ) %>% 
  arrange(desc(balance))


# DATA FORMAT TO PERFORM PLOT
emoji_feeling_user %>% 
  mutate( negative  = -negative,
          neutral.positive =  neutral/2,
          neutral.negative = -neutral/2) %>% 
  select(-neutral) %>% 
  gather("sentiment","mean", -author, -balance) %>% 
  mutate(sentiment = factor(sentiment, levels = c("negative", "neutral.negative", "positive", "neutral.positive"), ordered = T)) %>% 
  ggplot(aes(x=reorder(author,balance), y=mean, fill=sentiment)) +
  geom_bar(position="stack", stat="identity", show.legend = F, width = .5) +
  scale_fill_manual(values = brewer.pal(4,"RdYlGn")[c(1,2,4,2)]) +
  ylab(" - Negative / Neutral / Positive +") + xlab("User") +
  ggtitle("Sentiment analysis per user ","Based on average emoji sentiment score") +
  coord_flip() +
  theme_minimal() 

ggplotly()





# MOST FREQUENT EMOTION
# REMOVE EMOJIS
emoji_chat <- miChat %>% 
  unnest(emoji, emoji_name) %>% 
  mutate( emoji = str_sub(emoji, end = 1)) %>%  # REMOVE LINKS
  mutate( emoji_name = str_remove(emoji_name, ":.*")) # REMOVE LIGATURE NAMES

# TOKENIZE EMOJI'S NAME
emoji_chat <- emoji_chat %>% 
  select(author, emoji_name) %>% 
  unnest_tokens(input=emoji_name, output=emoji_words)

# GET ANOTHER LEXICON WITH NAME OF FEELINGS
lexical_sentimientos <- get_sentiments("nrc") # NAME OF FEELING
# FEELINGS PREVIEW
lexical_sentimientos %>% 
  head(10) %>% 
  kable() %>%
  kable_styling(full_width = F, font_size = 11)

# JOIN WITH EMOJIS 
feeling_chat <- emoji_chat %>% 
  inner_join(lexical_sentimientos, by=c("emoji_words"="word")) %>% 
  filter(!sentiment %in% c("negative","positive")) # REMOVE POITIVE / NEGATIVE CLASSIFICATION

# REMOVE EMOJIS
emoji_emocion <- miChat %>%
    select( emoji, emoji_name) %>% 
    unnest( emoji, emoji_name) %>% 
    mutate( emoji = str_sub(emoji, end = 1)) %>%  # REMOVE LINKS
    mutate( emoji_name = str_remove(emoji_name, ":.*")) %>%  # REMOVE LIGATURE NAMES
    unnest_tokens(input=emoji_name, output=emoji_words) %>% 
    inner_join(lexical_sentimientos, by=c("emoji_words"="word")) %>% 
    filter(!sentiment %in% c("negative","positive")) %>% # REMOVE NEGATIVE / POSITIVE RATING
    # KEEP ONLY THE 4 MOST FREQUENT EMOJI FOR EACH FEELING
    count(emoji, emoji_words, sentiment) %>% 
    group_by(sentiment) %>% 
    top_n(4,n) %>% 
    slice(1:4) %>% 
    ungroup() %>% 
    select(-n)



# PLOT OF MOSTLY EXPRESSED EMOTIONS
feeling_chat %>% 
  count(sentiment) %>% 
  ggplot(aes(x=reorder(sentiment,n), y=n)) +
  geom_col(aes(fill=n), show.legend = FALSE, width = .1) +
  geom_point(aes(color=n), show.legend = FALSE, size = 3) +
  coord_flip() +
  ylab("Number of times expressed") + xlab("Emotion") +
  scale_fill_gradient(low="#2b83ba",high="#d7191c") +
  scale_color_gradient(low="#2b83ba",high="#d7191c") +
  ggtitle("Most frequently expressed emotion","Expressed by use of emojis") +
  theme_minimal()

ggplotly()





# Most Frequent Feeling Per User

# PLOT OF EMOTIONS PER USER
feeling_chat %>% 
    count(author, sentiment) %>% 
    left_join(filter(lexical_sentimientos, sentiment %in% c("negative","positive")),by=c("sentiment"="word")) %>% 
    rename( feeling = sentiment.y) %>% 
    mutate( feeling = ifelse(is.na(feeling), "neutral", feeling)) %>% 
    mutate( feeling = factor(feeling, levels = c("negative", "neutral", "positive"), ordered=T) ) %>% 
    group_by(author) %>%
    top_n(n = 8, n) %>%
    slice(1:8) %>% 
    ggplot(aes(x = reorder(sentiment, n), y = n, fill = feeling)) +
    geom_col() +
    scale_fill_manual(values = c("#d7191c","#fdae61", "#1a9641")) +
    ylab("Number of times expressed") +
    xlab("Emotion") +
    coord_flip() +
    facet_wrap(~author, ncol = 6, scales = "free_x") +
    ggtitle("Emotions mostly expressed by user", "Expressed by use of emojis") + 
    theme_minimal() + theme(legend.position = "bottom")


ggplotly()



