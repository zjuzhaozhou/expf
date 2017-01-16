Expert finding for Community Question Answering
===

Code based on [theano tutorial](http://deeplearning.net/tutorial/lstm.html)

## Usage

Run data_engine.py to generate data, then fed generated pickle file to model.py and train model.

### Input
- **sample_i2q**: two columns, (question_id, normalized question text)
- **sample_i2t**: two columns, (user_id, user_twitter_id)
- **sample_twitter_links**: each row represents a twitter user and its following users, sperated by a colon, every id in the row is a twitter id
- **sample_q2u2s**: three columns, (question_id, user_id, rating)

### Output

- **`*`_graph.`*`**: deepwalk edgelist input, refer to [deepwalk code](https://github.com/phanein/deepwalk) for description
- **`*`_paths.`*`**: deepwalk path samples output, for debug purpose
- **`*`_expf.pkl**: pickle file of `train`, `train_for_test, valid`, `test` datasets. `train` is list of triplets, each element of the triplet can be represented as (question, user_pos, user_neg). `train_for_test`, `valid`, `test` datasets are for testing purpose, each dataset is organized as (questions list, users list, expert users list). Each element with same index of these three lists corresponds to a test sample (question, users answered the questions, expert users). `question` is represented as list of word indices.

**Read the Code for detailed representations.**
