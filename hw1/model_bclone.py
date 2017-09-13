import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import load_policy
import gym


flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("num_rollouts", 10, "Number of rollouts during evaluation [10]")
flags.DEFINE_boolean("is_render", False, "True to render during test [False]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
flags.DEFINE_integer("epoch", 1000, "Number of training iterations [2000]")
flags.DEFINE_string("log_dir", "./log/", "Number of training epochs [100]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Number of training epochs [100]")
flags.DEFINE_integer("save_epoch", 100, "Save checkpoints per n epoch [100]")
flags.DEFINE_integer("depth", 4, "Depth of fully connected neural network [4]")
flags.DEFINE_integer("width", 50, "Width of fully connected neural network [10]")
flags.DEFINE_float("gaussian_var", 0.1, "Gaussian variance for guassian policy [.1]")
flags.DEFINE_integer("batch_size", 1000, "Batch size for training neural network [100]")
flags.DEFINE_integer("n_demo", 10000, "Number of demonstrations for training data [10000]")
flags.DEFINE_string("env_name", "Hopper-v1", "name of gym environment")
flags.DEFINE_string("expert_policy_file", "expert/Hopper-v1.pkl", "Path to expert file")

FLAGS = flags.FLAGS


class NeuralNet(object):
    def __init__(self, sess, depth=4, width=10, learning_rate=0.001, epoch=2000, gaussian_var=.1, is_train=True,
                 log_dir="./log/", checkpoint_dir="./checkpoint/", save_epoch=100, batch_size=100, n_demo=10000,
                 env_name="Hopper-v1", expert_policy_file="expert/Hopper_v1.pkl", num_rollouts=10, is_render=False):
        assert(depth >= 1)  # at least one hidden layer
        self.sess = sess
        self.depth = depth
        self.width = width
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.gaussian_var = gaussian_var
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.is_train = is_train
        self.batch_size = batch_size
        self.n_demo = n_demo
        self.env_name = env_name
        self.expert_policy_file = expert_policy_file
        self.num_rollouts = num_rollouts
        self.is_render = is_render
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # environment specific options
        self.policy_fn = load_policy.load_policy(self.expert_policy_file)
        self.env = gym.make(self.env_name)

        self.x_dim, self.y_dim = self.get_dims()
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.x_dim], name="x")
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="y")
        y_nn = self.fc_network(self.x)

        self.l2_loss = tf.nn.l2_loss(y_nn-self.y) / self.batch_size
        self.l2_sum = tf.summary.scalar("l2_loss", self.l2_loss)
        self.saver = tf.train.Saver()

    def fc_network(self, x, reuse=False):
        with tf.variable_scope("fc_network") as scope:
            if reuse:
                scope.reuse_variables()
            h = x
            # hidden layers
            for l in range(self.depth):
                h = self.dense_bn_relu(h, scope="hidden_{0}".format(l + 1))
            # last layer
            y_nn = self.dense(h, output_dim=self.y_dim, scope="last_layer")

        return y_nn

    def train(self):
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.l2_loss)
        tf.global_variables_initializer().run()
        counter = 0
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data_fname = self.env_name + "_data.npz"
        data_fpath = os.path.join(self.checkpoint_dir, data_fname)
        # load data file if exists
        if os.path.exists(data_fpath):
            print("Loading data from data file...")
            demo_data = np.load(data_fpath)
            x_all, y_all = demo_data["x_all"], demo_data["y_all"]
        else:
            print("Performing rollouts to generate training data...")
            x_all, y_all = self.get_data()
            x_all, y_all = np.squeeze(x_all), np.squeeze(y_all)
            print("Saving data to data file...")
            np.savez(data_fpath, x_all=x_all, y_all=y_all)

        print("Training...")
        for it in tqdm(range(self.epoch)):
            # shuffle
            perm = np.random.permutation(self.n_demo)
            x_all, y_all = x_all[perm], y_all[perm]

            # split into mini-batches
            batch_idxs = self.n_demo // self.batch_size
            for idx in range(batch_idxs):
                x_batch = x_all[idx*self.batch_size:(idx+1)*self.batch_size, :]
                y_batch = y_all[idx*self.batch_size:(idx+1)*self.batch_size, :]
                _, summary_str = self.sess.run([train_op, self.l2_sum],
                                               feed_dict={self.x: x_batch,
                                                          self.y: y_batch})
                self.writer.add_summary(summary_str, counter)
                counter += 1

            if np.mod(it, self.save_epoch) == 0:
                print("[Save] Saving checkpoints at iter {0}".format(it))
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "nn.model"), it)
        # save final checkpoint
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "nn.model"), counter)

    def test(self):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        assert could_load
        # build graph for evaluation
        x_eval = tf.placeholder(tf.float32, [1, self.x_dim])
        y_mean = self.fc_network(x_eval, reuse=True)
        # gaussian policy
        y_std = self.gaussian_var * tf.ones(self.y_dim, dtype=tf.float32)
        y_dist = tf.contrib.distributions.MultivariateNormalDiag(y_mean, y_std)
        y_eval = y_dist.sample()

        # perform rollout
        print("Testing. Performing rollouts.")
        returns = []
        observations = []
        actions = []

        for i in tqdm(range(self.num_rollouts)):
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = self.sess.run(y_eval, feed_dict={x_eval: obs[None, :]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.is_render:
                    self.env.render()  # render results
                if steps >= self.env.spec.timestep_limit:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    def get_data(self):
        max_steps = self.env.spec.timestep_limit
        observations = []
        actions = []

        nobs = 0
        pbar = tqdm(total=self.n_demo)

        while nobs < self.n_demo:
            # get policy rollout sample
            obs = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = self.policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, _, done, _ = self.env.step(action)
                steps += 1
                if steps >= max_steps:
                    break
            pbar.update(len(observations) - nobs)
            nobs = len(observations)
        pbar.close()

        return np.array(observations[:self.n_demo]), np.array(actions[:self.n_demo])

    def get_dims(self):
        obs_dim, act_dim = self.env.reset().shape[0], self.env.action_space.sample().shape[0]
        return obs_dim, act_dim

    # ops
    def dense_bn_relu(self, input, scope):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(
                input, self.width, activation_fn=None, scope="dense")
            h2 = tf.contrib.layers.batch_norm(
                h1, center=True, scale=True, updates_collections=None, is_training=self.is_train, scope="bn")
        return tf.nn.relu(h2)

    def dense(self, input, output_dim, scope):
        with tf.variable_scope(scope):
            h = tf.contrib.layers.fully_connected(
                input, num_outputs=output_dim, activation_fn=None, scope="dense")
        return h

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


def main():
    # create folders if they don't exist
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    # check that expert file exists
    print(FLAGS.expert_policy_file)
    assert(os.path.exists(FLAGS.expert_policy_file))

    with tf.Session() as sess:
        neural_net = NeuralNet(
            sess=sess,
            is_train=FLAGS.is_train,
            depth=FLAGS.depth,
            width=FLAGS.width,
            gaussian_var=FLAGS.gaussian_var,
            learning_rate=FLAGS.learning_rate,
            epoch=FLAGS.epoch,
            log_dir=FLAGS.log_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_epoch=FLAGS.save_epoch,
            batch_size=FLAGS.batch_size,
            n_demo=FLAGS.n_demo,
            num_rollouts=FLAGS.num_rollouts,
            is_render=FLAGS.is_render,
            env_name=FLAGS.env_name,
            expert_policy_file=FLAGS.expert_policy_file
        )
        if FLAGS.is_train:
            neural_net.train()
        else:
            neural_net.test()


if __name__ == '__main__':
    main()
