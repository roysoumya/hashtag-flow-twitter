import sys
import tensorflow as tf
import numpy as np
import math
from tensorflow.python import debug as tf_debug

processed = sys.argv[1]
rank=sys.argv[2]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

with open(processed, "r") as f:
	content = f.readlines()
	f.close()
with open(rank, "r") as f2:
	content2 = f2.readlines()
	f2.close()

intervals=[]

for line in content2:
	tokens=line.split("\t")
	t1=float(tokens[0])/1000000
	t2=float(tokens[1])/1000000
	intervals.append((t1,t2))
L=len(intervals)
p=len(content)/7
counts=[[0]*(L-1)]*p
testcounts=[[0]*2]*p

test_t=[]
kh=[0 for x in range(p)]
v=[0 for x in range(p)]
w=[0 for x in range(p)]
b_lambda=[0 for x in range(p)]

l=0
h_series = [[] for x in range(p)]
kh_series = [[] for x in range(p)]
integral = [[] for x in range(p)]

num_epochs = 100
truncated_backprop_length = 15
echo_step = 3
batch_size = 5
expmod = 1000

batchk_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batcht_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
	
init_h=tf.placeholder(tf.float32, (batch_size))
init_kh=tf.placeholder(tf.float32, (batch_size))

for n in range(p):
#for n in range(1):
	
		
	v[n]=tf.Variable([0.001 for x in range(batch_size)])
	w[n]=tf.Variable([0.001 for x in range(batch_size)])
	b_lambda[n]=tf.Variable([0.001 for x in range(batch_size)])
	beta=tf.Variable([0.001 for x in range(batch_size)])
	gamma=tf.Variable([0.001 for x in range(batch_size)])
	bh=tf.Variable([0.001 for x in range(batch_size)])
	
	omega = tf.constant([0.01], shape=[batch_size])
	
	str_k=content[l]
	k=[float(ki) for ki in str_k.split(",")]
	str_train=content[l+1].rstrip()
	train_t=[float(ti)/1000000 for ti in str_train.split(",")] 
	str_test=content[l+6].rstrip()
	test_t=[float(ti)/1000000 for ti in str_test.split(",")]
	l=l+7
		
	for tj in train_t:
		for i in range(L-1):
			t1,t2=intervals[i]
			if t1<=tj and tj<=t2:
				counts[n][i]=counts[n][i]+1
				break;
	for tj in test_t:
		t1,t2=intervals[L-2]
		if t1<=tj and tj<=t2:
			testcounts[n][0]=testcounts[n][0]+1
			
		t1,t2=intervals[L-1]
		if t1<=tj and tj<=t2:
			testcounts[n][1]=testcounts[n][1]+1
			
	
	total_series_length = len(k)
	num_batches = total_series_length//batch_size//truncated_backprop_length
	
	
	k_series = tf.unstack(batchk_placeholder, axis=1)
	t_series = tf.unstack(batcht_placeholder, axis=1)
	
	
	# Forward pass
	current_h = init_h
	current_kh = init_kh
	for i in range(len(k_series)):
		current_k = k_series[i]
		current_t = t_series[i]
		
		#kernel = any other function yields NaNs as well. So it is not the kernel's fault
		
		#kernel = tf.exp(tf.truediv(tf.multiply(omega,current_t), tf.multiply(current_kh, current_k))) #yields NaNs
		#next_h = tf.add(tf.add(tf.multiply(beta,kernel),tf.multiply(gamma,current_h)),bh)
		#next_kh = tf.exp(tf.multiply(v[n],next_h))
		
		numerator = omega*current_t
		denominator = current_kh*current_k
		frac = tf.truediv(numerator,denominator)
		kernel = tf.exp(frac/expmod) #yields NaNs
		next_h = tf.exp((beta*kernel + gamma*current_h + bh)/expmod)
		#next_kh = tf.exp(v[n]*next_h)
		next_kh = v[n]*next_h
		h_series[n].append(next_h)
		kh_series[n].append(next_kh)
		current_h = next_h
		current_kh = next_kh
	
	
	reinforcement_loss = -tf.reduce_sum(tf.exp((v[n]*h_series[n][-1]+w[n]*(t_series[-1]-t_series[0])+b_lambda[n])/expmod))
		
	competition_error=[]
	for i in range(len(intervals)-1):
		t1, t2 = intervals[i]
		integral[n].append(expmod*tf.exp((v[n][-1]*h_series[n][-1][-1]-w[n][-1]*train_t[-1]+b_lambda[n][-1])/expmod)*expmod*(tf.exp((w[n][-1]*t2)/expmod)-tf.exp((w[n][-1]*t1)/expmod))/w[n][-1])
		competition_error.append(tf.square(integral[n][i]-counts[n][i]))
	
	competition_loss = tf.reduce_sum(competition_error)
	regularizer = tf.add_n([tf.nn.l2_loss(v[n]),tf.nn.l2_loss(w[n]),tf.nn.l2_loss(b_lambda[n]),tf.nn.l2_loss(beta),tf.nn.l2_loss(gamma),tf.nn.l2_loss(bh)])
	
	total_loss = reinforcement_loss+regularizer+0.003*competition_loss
	
	train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		loss_list = []
		h_list = []
		kh_list = []
		last_h = np.ones((batch_size))
		last_kh = np.ones((batch_size))
		last_batchk = np.ones((batch_size, truncated_backprop_length))
		last_batcht = np.ones((batch_size, truncated_backprop_length))
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		#writer = tf.summary.FileWriter("/home/supritam/Documents/aaai/code", sess.graph)

    	
		for epoch_idx in range(num_epochs):
			_current_h = np.ones((batch_size))
			_current_kh = np.ones((batch_size))
        		        		
			#print("New data, epoch", epoch_idx)

			for batch_idx in range(num_batches):
				start_idx = batch_idx * batch_size * truncated_backprop_length
				end_idx = start_idx + batch_size * truncated_backprop_length
            			
				batchk = list(chunks(k[start_idx:end_idx], truncated_backprop_length))
				batcht = list(chunks(train_t[start_idx:end_idx], truncated_backprop_length))
				
				#---debugging statements---
				#print batchk, _current_h
								            			
				_v, _w, _b_lambda, _total_loss, _train_step, _current_h, _current_kh = sess.run(
                	[v[n], w[n], b_lambda[n], total_loss, train_step, current_h, current_kh],
                	feed_dict={
						batchk_placeholder:batchk,
						batcht_placeholder:batcht,
						init_h:_current_h,
						init_kh:_current_kh
				})
				
				last_h = _current_h
				last_kh = _current_kh
				last_batchk = batchk
				last_batcht = batcht

				loss_list.append(_total_loss)
				h_list.append(_current_h)
				kh_list.append(_current_kh)
				#if batch_idx%100 == 0:
				#print("Step", batch_idx, "Loss", _total_loss)
				#print("v", _v)
				#print("w", _w)
				#print("b_lambda", _b_lambda)
		
		
		
		
		saved_v, saved_w, saved_b_lambda = sess.run([v[n], w[n], b_lambda[n]], feed_dict={
                    	batchk_placeholder:last_batchk,
                    	batcht_placeholder:last_batcht,
                    	init_h:last_h,
                    	init_kh:last_kh
                })
		t1,t2=intervals[L-2]
		testintegral = expmod*math.exp((saved_v[-1]*last_h[-1]-saved_w[-1]*train_t[-1]+saved_b_lambda[-1])/expmod)*expmod*(math.exp((saved_w[-1]*t2)/expmod)-math.exp((saved_w[-1]*t1)/expmod))/saved_w[-1]
		print t1, t2, "Predicted for 2nd last interval:", testintegral, "Actual for 2nd last interval:", testcounts[n][0]
		t1,t2=intervals[L-1]
		testintegral = expmod*math.exp((saved_v[-1]*last_h[-1]-saved_w[-1]*train_t[-1]+saved_b_lambda[-1])/expmod)*expmod*(math.exp((saved_w[-1]*t2)/expmod)-math.exp((saved_w[-1]*t1)/expmod))/saved_w[-1]
		print t1, t2, "Predicted for last interval:", testintegral, "Actual for last interval:", testcounts[n][1]
		#---debugging statements---
		#print h_list[:10]
		#print kh_list[:10]
		
		with open("parameter_output.txt", "a") as out:
			out.write("v: "+str(saved_v)+
			"\tw: "+str(saved_w)+
			"\tb_lambda: "+str(saved_b_lambda)+
			"\tlast_h: "+str(last_h)+"\tlast_kh: "+str(last_kh)+"\tt_f: "+str(train_t[-1])+"\n")
			out.close() 
		

	
	
	
