LJ40K
=====

Python modules for analyzing LJ40K emotion data

## System flow

![feelit flow](https://cloud.githubusercontent.com/assets/1659204/5698196/fd3873e8-9a42-11e4-803e-81c59a12c143.png)

## Training: batch/batchSimpleTrain.py

perform SVM training for LJ40K

1. usage
	
	```
	batchSimpleTraining.py [-h] [-k NFOLD] [-o OUTPUT_NAME] 
							[-e EMOTION_IDS] [-c C] [-g GAMMA] [-t TEMP_DIR] 
							[-n] [-v] [-d] 
							feature_list_file
	
	positional arguments:
  		feature_list_file   This program will fuse the features listed in this
                        	file and feed all of them to the classifier. The file
                        	format is in JSON. See "feautre_list_ex.json" for
                        	example

	optional arguments:
		-h, --help          show this help message and exit
  		-k NFOLD, --kfold NFOLD
                        	k for kfold cross-validtion. If the value less than 2,
                        	we skip the cross-validation and choose the first
                        	parameter of -c and -g (DEFAULT: 10)
  		-o OUTPUT_NAME, --output_file_name OUTPUT_NAME
                        	path to the output file in csv format (DEFAULT:
                        	out.csv)
  		-e EMOTION_IDS, --emotion_ids EMOTION_IDS
                        	a list that contains emotion ids ranged from 0-39
                        	(DEFAULT: 0). This can be a range expression, e.g.,
                        	3-6,7,8,10-15
  		-c C                SVM parameter (DEFAULT: 1). This can be a list
                        	expression, e.g., 0.1,1,10,100
  		-g GAMMA, --gamma GAMMA
                        	RBF parameter (DEFAULT: 1/dimensions). This can be a
                        	list expression, e.g., 0.1,1,10,100
  		-t TEMP_DIR, --temp_output_dir TEMP_DIR
                        	output intermediate data of each emotion in the
                        	specified directory (DEFAULT: not output)
		-n, --no_scaling      do not perform feature scaling (DEFAULT: False)       
  		-v, --verbose       show messages
  		-d, --debug         show debug messages
 	```
 	
2. notes

	* The example file resides in batch/feature_list_ex.json
	  feature_list_file is in JSON format. Here is an example:
		
		```
	 	[
	    	{
	        	"feature": "TFIDF_TSVD300",
	            "train_dir": "adir/bdir",
	            "test_file": "cdir/ddir/TFIDF_TSVD.test.npz"
	        },
	        {
	            "feature": "keyword",
	            "train_dir": "adir/bdir",
	            "test_file": "cdir/ddir/keyword.test.npz"
			}
	    ]
		```
		
    * Use example:
    
    	```
    	python batchSimpleTraining.py -k 10 -e 0-39 -o output.csv -c 1,10,100,1000 -v feature_list_ex.json
    	python batchSimpleTraining.py -k 10 -e 0-39 -o output.csv -c 10,30,70,100,300,700,1000 -g 0.0001,0.0003,0.001,0.003,0.01,0.1 -t temp_dir -v TFIDF_TSVD300.json

		```

## Data: example script for generating 'pattern40'

"pattern40" is the data that sum up the personal event arrays for each sample.
The following script will fetch data from a MongoDb and save them into the input format of our training program.

	>> python batchFetchPatterns.py ~/projects/data/MKLv2/2000samples_4/pattern40_all.npz
  	>> python batchSplitEmotion.py -b 0 -e 800 -p random_idx.pkl -s -x .train.npz -d ~/projects/data/MKLv2/2000samples    _4/pattern40_all.npz ~/projects/data/MKLv2/2000samples_4/train/pattern40/800p800n_Xy/pattern40.800p800n_Xy
  	>> python batchSplitEmotion.py -b 800 -e 1000 -d ~/projects/data/MKLv2/2000samples_4/pattern40_all.npz ~/projects/data/MKLv2/2000samples_4/test_8000/pattern40/full.Xy/pattern40.full.Xy.test.npz

		
## Programming: feelit/features.py

1. Load features from files

	```python
	>> from feelit.features import LoadFile
	>> lf = LoadFile(verbose=True)
	>> lf.loads(root="../emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=800)
	>> lf.dump(path="data/image_rgb_gist.Xy", ext=".npz")
	```
2. Load features from mongodb

	```python
	>> from feelit.features import FetchMongo
	>> fm = FetchMongo(verbose=True)
	>> fm.fetch_transform('TFIDF', '53a1921a3681df411cdf9f38', data_range=800)
	>> fm.dump(path="data/TFIDF.Xy", ext=".npz")
	```

3. Fuse loaded features

	```python
	>> from feelit.features import Fusion
	>> fu = Fusion(verbose=True)
	>> fu.loads(a1, a2, ...)
	>> fu.fuse()
	>> fu.dump()
	```
	
4. Train, Cross-validation and Test

	```python
	>> from feelit.features import Learning
	>> learner = Learning(verbose=args.verbose, debug=args.debug) 
    >> learner.set(X_train, y_train, feature_name)
    >>
    >> scores = {}
    >> for C in Cs:
    >> 	for gamma in gammas:
    >> 		score = learner.kFold(kfolder, classifier='SVM', 
    >>							kernel='rbf', prob=False, 
    >>							C=c, scaling=True, gamma=gamma)
    >>		scores.update({(c, gamma): score})
    >>
	>> best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]
	>> learner.train(classifier='SVM', kernel='rbf', prob=True, C=best_C, gamma=best_gamma, 
	>>				scaling=True, random_state=np.random.RandomState(0))
	>> results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)
	```
