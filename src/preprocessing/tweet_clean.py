import pandas as pd
import re
# remove @user, # and link 
class TweetClean():
    
    def transform(self, X_df: pd.DataFrame):  
        tweet_reg = ["tweet"]
        df = X_df
        for col in tweet_reg:
            #remove text after # with space at the end
            df[col] = df[col].apply(lambda x : re.sub("#[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+\s","",x))
            #remove text starting with # and at the end of sentence
            df[col] = df[col].apply(lambda x : re.sub("#[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+","",x))
            #remove text after @ with space at the end
            df[col] = df[col].apply(lambda x : re.sub("@[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+\s+","",x))
            #remove text starting with @ and at the end of sentence
            df[col] = df[col].apply(lambda x : re.sub("@[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+","",x))
            df[col] = df[col].apply(lambda x : re.sub(r'(\s)https?:\/\/.*[\r\n]*',r'',x))
            # remove all non alphabet and nun number to remove emojis encluding punctuation
            # we will not be needing the punctuation remover after this
            df[col] = df[col].apply(lambda x : re.sub("[^a-zA-Z0-9 ]+","",x))
        return df

