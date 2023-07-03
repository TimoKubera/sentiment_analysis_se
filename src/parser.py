import argparse
import os

parser = argparse.ArgumentParser(
                    prog='Sentiment Analysis',
                    description='todo: What does the program do?',
                    epilog='todo: Text at the bottom of the help dialoge')


parser.add_argument('-f', '--filename', required=False)         # training data
parser.add_argument('-m', '--model', required=False)            # User can choose to run a particular machine learning model

args = parser.parse_args()