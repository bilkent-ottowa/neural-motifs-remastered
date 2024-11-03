from data.collect_carla_bev import RGB_Cam, Semantic_Cam, ActorCar, CarlaBEVSampler

import carla
import argparse
import traceback

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        default=False,
        type=bool,
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--numBurn',
        metavar='N',
        default=0,
        type=int,
        help='Number of burn ticks (default: 0)')
    argparser.add_argument(
        '--save-dir',
        metavar='D',
        default='data/carla_bev',
        help='Directory to save the data (default: data/carla_bev)')
    argparser.add_argument(
        '--save-name',
        metavar='N',
        default='carla_bev.pkl',
        help='Name of the file to save the data (default: carla_bev.pkl)')
    argparser.add_argument(
        '--numTicks',
        default=1000,
        type=int,
        help='Number of ticks to run the simulation (default: 1000)')

    args = argparser.parse_args()

    # Connect to the CARLA server
    client = carla.Client(args.host, args.port)
    


    # Initialize the CarlaBEVSampler object
    sampler = CarlaBEVSampler(client, 
                              tmPort = args.tm_port, 
                              mode = 'async' if args.asynch else 'sync', 
                              numBurn = args.numBurn, 
                              save_dir = args.save_dir, 
                              save_name = args.save_name)
    

    # Start the data collection
    sampler.run(args.numTicks)

    # Save the data
    sampler.save()

    return 1

if __name__ == '__main__':  
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    except Exception as e:
        print(e)
        traceback.print_exc()