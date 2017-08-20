import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import tweepy
import os
import json
import random
import sys

apilist=[]
accselect=0

def get_auth(acc_details):

    auth = tweepy.OAuthHandler(acc_details['consumer_key'], acc_details['consumer_secret'])
    auth.set_access_token(acc_details['access_token'], acc_details['access_token_secret'])
    return auth

def get_api(acc_details):

    api = tweepy.OAuthHandler(acc_details['consumer_key'], acc_details['consumer_secret'])
    api.set_access_token(acc_details['access_token'], acc_details['access_token_secret'])
    return tweepy.API(api)

def readjson():
    accfile='./accts.json'
    try:
        with open(accfile, 'r') as account_data:
            d = json.load(account_data)
            for item in d:
                duplicate=0
                for api in apilist:
                    if(api["consumer_key"] == item["consumer_key"] and api["consumer_secret"] == item["consumer_secret"] and api["access_token"] == item["access_token"] and api["access_token_secret"] == item["access_token_secret"]):
                        duplicate=1
                if(duplicate==0):
                    apilist.append(item)
                else:
                    print("Duplicate Account Read & Ignored")
    except:
        print("No json Account file found.")

def writejson():
    with open("./accts.json", "w") as outfile:
        json.dump(apilist, outfile)

def initaccs():
    readjson()
    path='./input/'
    namearray=[]
    try:
        for filename in os.listdir(path):
            namearray.append(path+filename)
        for name in namearray:
            try:
                f = open(name)
        
                store = {}
                store["acc_name"]=name[8:] #removes directory preamble
                store["consumer_key"]=f.readline()[:-1] #removes the \n
                store["consumer_secret"]=f.readline()[:-1]
                store["access_token"]=f.readline()[:-1]
                store["access_token_secret"]=f.readline()[:-1]
                duplicate=0
                for api in apilist:
                    if(api["consumer_key"] == store["consumer_key"] and api["consumer_secret"] == store["consumer_secret"] and api["access_token"] == store["access_token"] and api["access_token_secret"] == store["access_token_secret"]):
                    #if(api["consumer_key"] == store["consumer_key"]):
                        duplicate=1
                if(duplicate==0):
                    apilist.append(store)
                else:
                    print("Discarded duplicate account file")
            except:
                print("Cannot open: "+name)
        print("Found "+str(len(apilist))+" account file(s).")
        writejson()
        #print(apilist)
    except:
        print("Please enter your Twitter App Keys into four seperate lines in a text file in this order: \n 1.consumer key\n 2.consumer secret\n 3.access token\n 4.access token secret\n Place the text file into the input directory and rerun")

#function to pick active account
def pick_acc():

    accselmsg="Please enter a number corresponding to the account you with to use: \n"
    for i in range(0,len(apilist)):
        accdetail=str(i)+" - "+apilist[i]["acc_name"]+"\n"
        accselmsg+=accdetail
    print(accselmsg)
    accselect=int(input())
    print("Setting current account to "+apilist[accselect]["acc_name"])

def tensortrain():
    batch_size=128
    embedding_size=2
    trainingfile="manageddata.txt"

    def generate_lex(trainingfile):
        print("Beginning Lexicon Generation... ")
        datafile=open(trainingfile, "r")
        data=datafile.read()
        datafile.close
        indata=data.split()
        num_sampled=len(data.split())
        vocabulary=set(data.split())
        vocabulary_size=len(vocabulary)
        print("Vocabulary Size: "+str(vocabulary_size))
        word2num=dict((c,i) for i,c in enumerate(vocabulary))
        num2word=dict((i,c) for i,c in enumerate(vocabulary))
        return word2num,num2word,vocabulary_size

    def generate_batches(datafile,batchsize,offset=0):
        nonlocal embedding_size 
        nonlocal word2num
        nonlocal num2word
        fn=open(datafile, 'r')
        inputdata=[]
        if offset>0:
            for i in range(offset):
                fn.readline()
        for i in range(batch_size-1):
            inputdata.extend(fn.readline().split())
        skip=2
        embedded_words=[]
        word_labels=[]
        for i in range(0, len(inputdata)-embedding_size, skip):
            for j in range(i, i+embedding_size):
                ew=inputdata[j:j+embedding_size]
                embedded_words.append(ew)
            word_labels.append(inputdata[i+embedding_size])
    
        X = np.zeros((len(embedded_words), embedding_size, vocabulary_size))
        Y = np.zeros((len(embedded_words), vocabulary_size))
        for i, sect, in enumerate(embedded_words):
            for j, word, in enumerate(sect):
                X[i,j,word2num[word]]=1
            Y[i,word2num[word]]=1
        X,Y=shuffle_batches(X,Y)
        return X, Y

    def collect_batches(trainingfile):
        nonlocal batch_size
        fn=open(trainingfile, 'r')
        input_size=sum(1 for line in fn)
        print("Number of lines in file:",input_size)
        fn.close()
        batches_x=[]
        batches_y=[]
        numruns=round(input_size/batch_size)
        offset=0
        for i in range(0,numruns):
            col_x,col_y=generate_batches(trainingfile, batch_size, offset=offset)
            offset=offset+batch_size
            batches_x.extend(col_x)
            batches_y.extend(col_y)
        return batches_x, batches_y
   
    def shuffle_batches(xb,yb):
        for i in range(len(xb)):
            rand1=random.randint(0,len(xb)-1)
            rand2=random.randint(0,len(xb)-1)
            tempx=xb[rand1]
            tempy=yb[rand1]
            xb[rand1]=xb[rand2]
            yb[rand1]=yb[rand2]
            xb[rand2]=tempx
            yb[rand2]=tempy
        return xb, yb
        
    
    word2num,num2word,vocabulary_size=generate_lex(trainingfile)
    emb_words,labels=collect_batches(trainingfile)
    emb_words,labels=shuffle_batches(emb_words,labels)
    
    def rnn_model(x):
        x=tf.split(x,1)
        nonlocal vocabulary_size
        rnn_size=128
        layer = {'weights':tf.Variable(tf.random_normal([rnn_size, vocabulary_size])),
                 'biases': tf.Variable(tf.random_normal([vocabulary_size]))}
        lstm=rnn.BasicLSTMCell(rnn_size)
        outputs, states= rnn.static_rnn(lstm, x, dtype=tf.float32)
        outputs=outputs[0]
        output=tf.add(tf.matmul(outputs, layer['weights']), layer['biases'])
        return output
    
    hm_epochs=3
    x = tf.placeholder('float',[embedding_size, vocabulary_size])
    y = tf.placeholder('float',[vocabulary_size])
    global_step=tf.Variable(0,name='global_step', trainable=False)
    globalstepup=tf.assign_add(global_step, 1, name='increment')
    nextword=rnn_model(x)
    nextword=nextword[0]
    cost=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=nextword, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        checkpoint_directory='./ckpt/'
        if tf.gfile.Exists(checkpoint_directory):
            try:
                metag=checkpoint_directory+'model.meta'
                saver=tf.train.import_meta_graph(metag)
                print(tf.train.latest_checkpoint(checkpoint_directory))
                saver.restore(sess, save_path=checkpoint_directory+'model')
            except:
                print("Couldn't load checkpoint...")
                pass
        save_every=1000
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for step in range(len(emb_words)):
                sess.run(globalstepup)
                for wordvec in range(len(emb_words[step])):
                    epoch_x=emb_words[step]
                    epoch_y=labels[step]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                if(step%500==0):
                    print('Loss at Global Trainging Step {}: {}'.format(sess.run(global_step), c))
                if(step % 1000 == 0 and step != 0):
                    print("Training Loss for this Epoch at Step {} of {}: {}".format(step,len(emb_words),epoch_loss))
                if(step % save_every == 0):
                    saver.save(sess, checkpoint_directory + 'model')
            print('Epoch:'+str(epoch+1), 'completed out of', str(hm_epochs), 'loss:', str(epoch_loss))
            
            
            emb_words,labels=shuffle_batches(emb_words,labels)

def cleandata():
    print("Data Scrub Starting...")
    f=open("dataset1.txt","r")
    data=f.read()
    data=data.lower()
    f.close
    chars=sorted(list(set(data)))
    manychars=len(chars)
    print(chars)
    option=input("Count the number of characters on the front of the above list that you'd like to keep and enter that number: ")
    option=int(option)*-1
    for char in chars[option+1:manychars]:
        data=data.replace(char, "")
    newfile="manageddata.txt"
    fn=open(newfile, "w")
    fn.write(data)
    fn.close
    print("Data Scrub Complete!")

def scrapetweets():
    initaccs()
    auth = get_auth(apilist[accselect])
    wordlist=[]
    class DataScraperStreamListener(tweepy.StreamListener):
        count=0
        starttime=time.time()
        def on_status(self, status):
            #print("Tweet: "+status.text)
            #print("Name: "+str(status.user.name))
            #print("Verified?: " +str(status.user.verified))
            if(status.user.verified==True):
                DataScraperStreamListener.count+=1
                tweettime=time.time()
                print("Name: "+str(status.user.name))
                print("Verified?: " +str(status.user.verified)+"\n")
                twtsps=DataScraperStreamListener.count/(tweettime-DataScraperStreamListener.starttime)
                print("Verified Tweets collected per Second: "+str(twtsps)+"\n")
                nextline=status.text.replace("\n"," ")+"\n"
                nextline2=nextline.replace("&amp;","&")
                links=[]
                for word in nextline2.split():
                    if 'http' in word:
                        links.append(word)
                #print(links)
                nextline3=nextline2
                for t in links:
                    nextline3=nextline3.replace(t,"")
                    #print(nextline3)
                print("Tweet: "+nextline3)
                f=open("verifiedaccounts.txt","a")
                f.write(nextline3)
                f.close
                f=open("twtspslog.txt","a")
                f.write(str(twtsps)+'\n')
                f.close
        
        def on_error(self, status_code):
            print(status_code)

    twitterStream = tweepy.Stream(auth, DataScraperStreamListener())
    filter=twitterStream.filter(languages=["en"], track=wordlist)

def main():

    
    option = input('Enter \'scrape\' or \'train\' or \'datascrub\': ')
    if option == 'tweet':
        tweet = input('Enter your tweet\n')
        initaccs()
        pick_acc()
        api = get_api(apilist[accselect])
        api.update_status(status=tweet)
    elif option == 'datascrub':
        cleandata()
    elif option == 'scrape':
        scrapetweets()
    elif option == 'train':
        tensortrain()

if __name__ == "__main__":
  main()
