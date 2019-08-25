import argparse
import sys

import train
import test
import utils


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Trains and tests DQNsort")
    parser.add_argument('mode', help='Training or testing mode')

    parser.add_argument('--use_visdom', type=utils.str2bool,
                        help='Use visdom visualizer', default=True)
    parser.add_argument('--n_epoch', type=int,
                        help='Number of epochs', default=100)
    parser.add_argument('--n_iter', type=int,
                        help='Number of iterations per epoch', default=10000000)
    parser.add_argument('--update_rate', type=int,
                        help='Rate to update array in visdom', default=5)
    parser.add_argument('--save_rate', type=int,
                        help='Rate to save model', default=5)
    parser.add_argument('--discount', type=float,
                        help='Q-learning discount', default=0.99)
    parser.add_argument('--lr', type=float,
                        help='DQN learning rate', default=1e-4)
    parser.add_argument('--batch_size', type=int,
                        help='DQN batch size', default=32)
    parser.add_argument('--make_gif', type=utils.str2bool,
                        help='Generate gif on comparison', default=True)
    

    args = parser.parse_args()

    if args.mode == 'train':
        # List args
        print("---")
        for arg in vars(args):
            print("{:<20} : {}".format(arg, getattr(args, arg)))
        print("---")
        
        train.train(
            use_visdom=args.use_visdom,
            n_epoch=args.n_epoch,
            n_iter=args.n_iter,
            update_rate=args.update_rate,
            save_rate=args.save_rate,
            discount=args.discount,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    elif args.mode == 'test':
        test.test()
    elif args.mode == 'compare':
        test.test_compare(args.make_gif)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
