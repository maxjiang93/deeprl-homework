import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import load_policy
import gym
import glob


flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("num_rollouts", 10, "Number of rollouts during evaluation [10]")
flags.DEFINE_boolean("is_render", False, "True to render during test [False]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
flags.DEFINE_integer("epoch", 100, "Number of training iterations [2000]")
flags.DEFINE_string("log_dir", "./log/", "Number of training epochs [100]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Number of training epochs [100]")
flags.DEFINE_integer("save_epoch", 100, "Save checkpoints per n epoch [100]")
flags.DEFINE_integer("depth", 4, "Depth of fully connected neural network [4]")
flags.DEFINE_integer("width", 50, "Width of fully connected neural network [10]")
flags.DEFINE_integer("batch_size", 1000, "Batch size for training neural network [100]")
flags.DEFINE_integer("n_demo", 10000, "Number of demonstrations for training data [10000]")
flags.DEFINE_integer("m_demo", 10000, "Number of demonstrations for training data [10000]")
flags.DEFINE_string("env_name", "Hopper-v1", "name of gym environment")
flags.DEFINE_string("expert_policy_file", "expert/Hopper-v1.pkl", "Path to expert file")
flags.DEFINE_integer("dagger_iters", 10, "Number of dagger iters to take [10]")
flags.DEFINE_integer("dagger_init_iter", 0, "DAgger starting iteration [0]")

FLAGS = flags.FLAGS


class NeuralNet(object):
    def __init__(self, sess, depth=4, width=10, learning_rate=0.001, epoch=2000, gaussian_var=.1, is_train=True,
                 log_dir="./log/", checkpoint_dir="./checkpoint/", save_epoch=100, batch_size=100, n_demo=10000,
                 m_demo=10000, env_name="Hopper-v1", expert_policy_file="expert/Hopper_v1.pkl", num_rollouts=10,
                 is_render=False, dagger_init_iter=0):
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
        self.m_demo = m_demo
        self.env_name = env_name
        self.expert_policy_file = expert_policy_file
        self.num_rollouts = num_rollouts
        self.is_render = is_render
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.dagger_init_iter = dagger_init_iter
        self.dagger_iter = dagger_init_iter
        self.counter = 0

        # environment specific options
        self.policy_fn = load_policy.load_policy(self.expert_policy_file)
        self.env = gym.make(self.env_name)

        self.x_dim, self.y_dim = self.get_dims()
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.x_dim], name="x")
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="y")
        y_nn = self.fc_network(self.x, train=True)

        self.l2_loss = tf.nn.l2_loss(y_nn-self.y) / self.batch_size
        self.l2_sum = tf.summary.scalar("l2_loss", self.l2_loss)
        self.saver = tf.train.Saver()

    def fc_network(self, x, reuse=False, train=True):
        with tf.variable_scope("fc_network") as scope:
            if reuse:
                scope.reuse_variables()
            h = x
            # hidden layers
            for l in range(self.depth):
                h = self.dense_bn_relu(h, scope="hidden_{0}".format(l + 1), train=train)
            # last layer
            y_nn = self.dense(h, output_dim=self.y_dim, scope="last_layer")

        return y_nn

    def train(self):
        if self.dagger_iter == self.dagger_init_iter:
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.l2_loss)
            tf.global_variables_initializer().run()
            self.counter = 0
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                self.counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        data_fname = self.env_name + "_dagger_{0}.npz".format(self.dagger_iter)
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
                _, summary_str = self.sess.run([self.train_op, self.l2_sum],
                                               feed_dict={self.x: x_batch,
                                                          self.y: y_batch})
                self.writer.add_summary(summary_str, self.counter)
                self.counter += 1

            if np.mod(it, self.save_epoch) == 0 and not it == 0:
                print("[Save] Saving checkpoints at epoch {0}".format(it+self.dagger_iter*self.epoch))
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "nn.model"),
                                it+self.dagger_iter*self.epoch)
        # save final checkpoint
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "nn.model"),
                        self.epoch+self.dagger_iter*self.epoch)

    def test(self, scope="outer"):
        if scope == "outer":
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            assert could_load
        # build graph for evaluation
        x_eval = tf.placeholder(tf.float32, [1, self.x_dim])
        y_mean = self.fc_network(x_eval, reuse=True, train=False)
        # gaussian policy
        y_std = self.gaussian_var * tf.ones(self.y_dim, dtype=tf.float32)
        y_dist = tf.contrib.distributions.MultivariateNormalDiag(y_mean, y_std)
        y_eval = y_dist.sample()

        # perform rollout
        print("Testing. Performing rollouts.")
        returns = []
        observations = []
        actions = []

        for _ in tqdm(range(self.num_rollouts)):
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

        # print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        if scope == "inner":
            return np.mean(returns), np.std(returns)

    def dagger_step(self):
        # unroll to acquire observation data
        # build graph for evaluation
        x_eval = tf.placeholder(tf.float32, [1, self.x_dim])
        y_mean = self.fc_network(x_eval, reuse=True, train=False)
        # gaussian policy
        y_std = self.gaussian_var * tf.ones(self.y_dim, dtype=tf.float32)
        y_dist = tf.contrib.distributions.MultivariateNormalDiag(y_mean, y_std)
        y_eval = y_dist.sample()

        max_steps = self.env.spec.timestep_limit
        observations = []
        actions = []

        nobs = 0
        pbar = tqdm(total=self.m_demo)

        while nobs < self.m_demo:
            # get policy rollout sample
            obs = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = self.sess.run(y_eval, feed_dict={x_eval: obs[None, :]})
                observations.append(obs)
                actions.append(action)
                obs, _, done, _ = self.env.step(action)
                steps += 1
                if steps >= max_steps:
                    break
            pbar.update(min(len(observations) - nobs, self.m_demo - nobs))
            nobs = len(observations)
        pbar.close()

        new_observations = observations[:self.m_demo]
        new_actions = []
        for new_obs in new_observations:
            new_act = self.policy_fn(new_obs[None, :])
            new_actions.append(new_act)

        # read, merge, and write data file
        data_fpath = os.path.join(self.checkpoint_dir, self.env_name + "_dagger_{0}.npz".format(self.dagger_iter))
        old_data = np.load(data_fpath)
        x_old, y_old = old_data["x_all"], old_data["y_all"]
        x_new, y_new = np.array(new_observations), np.array(new_actions)
        x_new, y_new = np.squeeze(x_new), np.squeeze(y_new)
        x_all = np.concatenate((x_old, x_new), axis=0)
        y_all = np.concatenate((y_old, y_new), axis=0)

        # remove old file
        os.remove(data_fpath)

        # write new file
        self.dagger_iter += 1
        data_fpath = os.path.join(self.checkpoint_dir, self.env_name + "_dagger_{0}.npz".format(self.dagger_iter))
        np.savez(data_fpath, x_all=x_all, y_all=y_all)

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
            pbar.update(min(len(observations) - nobs, self.n_demo - nobs))
            nobs = len(observations)
        pbar.close()

        return np.array(observations[:self.n_demo]), np.array(actions[:self.n_demo])

    def get_dims(self):
        obs_dim, act_dim = self.env.reset().shape[0], self.env.action_space.sample().shape[0]
        return obs_dim, act_dim

    # ops
    def dense_bn_relu(self, input, scope, train):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(
                input, self.width, activation_fn=None, scope="dense")
            h2 = tf.contrib.layers.batch_norm(
                h1, center=True, scale=True, updates_collections=None, is_training=train, scope="bn")
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
            learning_rate=FLAGS.learning_rate,
            epoch=FLAGS.epoch,
            log_dir=FLAGS.log_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_epoch=FLAGS.save_epoch,
            batch_size=FLAGS.batch_size,
            n_demo=FLAGS.n_demo,
            m_demo=FLAGS.m_demo,
            num_rollouts=FLAGS.num_rollouts,
            is_render=FLAGS.is_render,
            env_name=FLAGS.env_name,
            expert_policy_file=FLAGS.expert_policy_file,
            dagger_init_iter=FLAGS.dagger_init_iter
        )
        if FLAGS.is_train:
            means, stdevs = [], []
            for iter in range(FLAGS.dagger_init_iter, FLAGS.dagger_init_iter + FLAGS.dagger_iters):
                print("DAgger training iteration: {0}".format(iter))
                neural_net.train()
                print("DAgger data gen iteration: {0}".format(iter))
                neural_net.dagger_step()
                print("DAgger testing  iteration: {0}".format(iter))
                mean, stdev = neural_net.test("inner")

                means.append(mean)
                stdevs.append(stdev)

            # save mean and std info
            means, stdevs = np.array(means), np.array(stdevs)
            print("Saving DAgger training portfolio...")
            np.savez("./dagger_training.npz", means=means, stdevs=stdevs)

        else:
            neural_net.test()


if __name__ == '__main__':
    main()
