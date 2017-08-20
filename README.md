# twitbot

twitbot is an all in one solution for scraping and processing tweets using natural language processing. 

with twitbot, you can:
    - scrape tweets off of twitter looking for keywords or account attributes
    - clean the scraped tweets
    - process the cleaned tweets through a recursive neural net to predict the next words in a seT

install directions:

get dependencies:

    $ pip install numpy
    $ pip install tensorflow
    $ pip install tweepy

clone the repository:

    $ git clone https://github.com/ediamondscience/twitbot

move to the install directory

    $ cd twitbot

setup accounts:

    -use your favorite text editor and place the API keys into a textfile in this order:
        consumer key
        consumer secret
        access token
        acess token secret
     -create a directory named 'input' in your install directory
     -place the text file in the input directory
 
the application should be ready to run at that point
