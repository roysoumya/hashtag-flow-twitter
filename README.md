# hashtag-flow-twitter
Contains the codebase for the Social Computing Term Project, Autumn 2017 IIT Kharagpur. The name of our project is "Prediction of hashtag flow in Twitter" and the team comprised of Avirup Saha, Soumyadeep Roy, Srinidhi Moodalagiri, Kushal Gaikwad, Abhay Shukla.  The final report is in the file "Final_Report_SocialCompTermProject_2017.pdf".


# Description
In general, the popularity of a hashtag depends on the two primary factors : 
1. hashtag tweet reinforcement
2. hashtag-hashtag competition

Through this term project, we propose the following approaches : 
1. Linear memory less (LMM) which is a novel approach that relies on hashtag-tweet reinforcement.
2. SeqGAN model (as described in Yu et. al. 2017) which explicitly models inter-hashtag competition.

# Code usage instructions
Fields are as follows:

Processed[i].txt

k_array [popularity count]
training timestamp (relative to the first occurrence)
omega
omega_0
alpha
initial_value(beta, lambda_0)
testtimestamp

rank[i].txt

ranked_list of hashtags in different time window
(you can split as many as you can)

name[i].txt
list of competing hashtags

# Coding environment
Python 3.6.4

# References:
Yu, L.; Zhang, W.; Wang, J.; and Yu, Y. 2017. Seqgan: Sequence generative adversarial nets with policy gradient. In AAAI, 2852–2858

