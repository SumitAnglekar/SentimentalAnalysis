import web

import get_twitter_data

import baseline_classifier, naive_bayes_classifier, max_entropy_classifier
import json, logging, html_helper

urls = (
    '/', 'index'
)

class index:
    def GET(self):
        query = web.ctx.get('query')
        html = html_helper.HTMLHelper()
        twitterData = get_twitter_data.TwitterData()
        if query:
            if(query[0] == '?'):
                query = query[1:]
            arr = query.split('&')
            logging.warning(arr)
            
            #default values
            time = 'daily'

            for item in arr:
                if 'keyword' in item:
                    keyword = item.split('=')[1]
                elif 'method' in item:
                    method = item.split('=')[1]
                elif 'time' in item:
                    time = item.split('=')[1]
            #end loop
                            
            if(method != 'baseline' and method != 'naivebayes' and method != 'maxentropy'):
                return html.getDefaultHTML(error=2)
            
            tweets = twitterData.getTwitterData(keyword, time)
            if(tweets):
                if(method == 'baseline'):
                    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time)
                    bc.classify()
                    return bc.getHTML()
                elif(method == 'naivebayes'):
                    trainingDataFile = 'training_neatfile_4.csv'               
                    classifierDumpFile = 'moviereviewtesting-pari.pickle'
                    #classifierDumpFile = 'naivebayes_trained_model.pickle'
                    trainingRequired = 0
                    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time, \
                                                  trainingDataFile, classifierDumpFile, trainingRequired)
                    nb.classify()
                    return nb.getHTML()
                elif(method == 'maxentropy'):
                    trainingDataFile = 'training_neatfile_4.csv'                
                    classifierDumpFile = 'maxent_trained_model.pickle'
                    trainingRequired = 0
                    maxent = max_entropy_classifier.MaxEntClassifier(tweets, keyword, time, trainingDataFile, classifierDumpFile, trainingRequired)
                    maxent.classify()
                    return maxent.getHTML()
            else:
                return html.getDefaultHTML(error=1)
        else:
            return html.getDefaultHTML()

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()