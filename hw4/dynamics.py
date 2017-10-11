import tensorflow as tf
import numpy as np

eps = 1e-9


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """

        self.sess = sess
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # unpack components of normalization
        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = normalization
        mean_obs = np.expand_dims(mean_obs, axis=0).astype(np.float32)
        std_obs = np.expand_dims(std_obs, axis=0).astype(np.float32)
        mean_deltas = np.expand_dims(mean_deltas, axis=0).astype(np.float32)
        std_deltas = np.expand_dims(std_deltas, axis=0).astype(np.float32)
        mean_action = np.expand_dims(mean_action, axis=0).astype(np.float32)
        std_action = np.expand_dims(std_action, axis=0).astype(np.float32)

        # construct computation graph (assuming continuous actions space)
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # placeholder for raw observations and actions
        self.ob_raw_ph = tf.placeholder(tf.float32, [batch_size, self.ob_dim], name='ob_norm')
        self.ac_raw_ph = tf.placeholder(tf.float32, [batch_size, self.ac_dim], name='ac_norm')
        self.dl_raw_ph = tf.placeholder(tf.float32, [batch_size, self.ob_dim], name='dl_norm')

        # normalize observations and actions
        self.ob_norm_ph = (self.ob_raw_ph - mean_obs) / (std_obs + eps)
        self.ac_norm_ph = (self.ac_raw_ph - mean_action) / (std_action + eps)
        self.dl_norm_ph = (self.dl_raw_ph - mean_deltas) / (std_deltas + eps)

        # feed inputs to neural network
        self.input_ph = tf.concat((self.ob_norm_ph, self.ac_norm_ph), axis=-1)
        self.pred_norm = build_mlp(input_placeholder=self.input_ph,
                                   output_size=self.ob_dim,
                                   scope='mlp',
                                   n_layers=n_layers,
                                   size=size,
                                   activation=activation,
                                   output_activation=output_activation)

        # define loss
        self.loss = tf.losses.mean_squared_error(self.dl_norm_ph, self.pred_norm)

        # un-normalized predictions
        self.pred_raw = mean_deltas + std_deltas * self.pred_norm

        # construct graph for training
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """

        # unpack data
        obs = []
        deltas = []
        action = []
        for i in range(len(data)):
            delta_i = [data[i]['next_observations'][j] - data[i]['observations'][j]
                       for j in range(len(data[i]['next_observations']))]
            obs.append(data[i]['observations'])
            action.append(data[i]['actions'])
            deltas.append(delta_i)

        obs, deltas, action = np.array(obs).reshape([-1, self.ob_dim]), \
                              np.array(deltas).reshape([-1, self.ob_dim]), \
                              np.array(action).reshape([-1, self.ac_dim])

        n_sample = obs.shape[0]

        # train for self.iterations epochs
        print("Fitting Dynamics Model...")
        from tqdm import tqdm

        for _ in tqdm(range(self.iterations)):
            # shuffle
            perm = np.random.permutation(n_sample)
            obs, deltas, action = obs[perm], deltas[perm], action[perm]

            # split into mini-batches
            batch_idxs = n_sample // self.batch_size - 1
            for idx in range(batch_idxs):
                ob_batch = obs[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                ac_batch = action[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                dl_batch = deltas[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                self.sess.run(self.training_op, feed_dict={self.ob_raw_ph: ob_batch,
                                                           self.ac_raw_ph: ac_batch,
                                                           self.dl_raw_ph: dl_batch})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        assert(states.shape[0] == actions.shape[0])
        n_sample = states.shape[0]
        batch_nopad = n_sample // self.batch_size
        pad_size = n_sample % self.batch_size
        preds = []

        for b in range(batch_nopad):
            states_batch = states[b * self.batch_size: (b+1) * self.batch_size]
            actions_batch = actions[b * self.batch_size: (b+1) * self.batch_size]
            pred_batch = self.sess.run(self.pred_raw, feed_dict={self.ob_raw_ph: states_batch,
                                                                 self.ac_raw_ph: actions_batch})
            preds.append(pred_batch)

        # pad array if needed
        if pad_size:
            b = batch_nopad
            states_batch = np.zeros([self.batch_size, self.ob_dim])
            actions_batch = np.zeros([self.batch_size, self.ac_dim])
            s0, s1 = states[b * self.batch_size:].shape
            a0, a1 = actions[b * self.batch_size:].shape
            states_batch[:s0, :s1] = states[b * self.batch_size:]
            actions_batch[:a0, :a1] = actions[b * self.batch_size:]
            pred_batch = self.sess.run(self.pred_raw, feed_dict={self.ob_raw_ph: states_batch,
                                                                 self.ac_raw_ph: actions_batch})
            preds.append(pred_batch[:s0])

        predictions = np.concatenate(preds, axis=0)

        return predictions