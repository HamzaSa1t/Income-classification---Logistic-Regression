This model predicts weather the person's income is 50k$ and higher or lower. it uses logistic regression model. a Classification problem.

	Steps:
	1- Data Cleaning:
			- Dropped: fnlwgt, education and marital-status columns
			- Replaced ? values with NaN, dropped rows missing workclass/occupation
			- Filled missing native-country using mode per race group
			
	2- Feature Engineering:
			- One-hot encoding
			- Grouped native-country into 4 buckets: US, Mexico, Developed, Developing
	
	3- Scaling
	
	4- Feature Importance Analysis
			- Computed correlation with target + permutation importance via sklearn
			- Dropped 7 near-zero/negative importance features
	
	5- model training:
			- trained the data 7 times:
							1- plain gradient descent
							2- Regularization — L2 penalty to reduce overfitting
							3- Polynomial features — degree-2 cross-terms on numerical features → small improvement
							4- Polynomial features with regularization
							5- Feature selection: dropped 7 low importance features + Log transform: capital gain and loss + Class weighting: upweighted positive class to improve recall
							6- same as 5 but with regularization
							7- using sklearn to measure my model's performance
	6- results:
			- the best custom model (Weighted Reg v2) which matches sklearn within ~0.7% accuracy and ~0.004 F1 — confirming the implementation is near perfect
							
