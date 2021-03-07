# Main function for the Tennis project if running from the command line.
# Edit the hyperparameters in here as desired, then run the program
# with no arguments.
#
# Running this function is identical to doing the training from the Jupyter
# notebook, but it does not attempt to plot the resulting scores.

def main():

    ##### set up key hyperparams

    RUN_NAME = 'TEST' #used for config control & naming of checpoint files
    BATCH = 512
    DECAY = 0.99999
    LEARN_EVERY = 10
    LEARN_ITER  = 4

    ##### instantiate the agents and perform the traing

    a = DdpgAgent(33, 4, random_seed=0, batch_size=BATCH, noise_decay=DECAY, 
                  learn_every=LEARN_EVERY, learn_iter=LEARN_ITER)

    scores = train(a, env, run_name=RUN_NAME, max_episodes=1000, max_time_steps=1000, 
                   break_in=BATCH, sleeping=True)


if __name__ == "__main__":
    main()

