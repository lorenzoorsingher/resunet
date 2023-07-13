import os
import argparse
from dotenv import load_dotenv
def parse_argv():
    load_dotenv()
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", help="Path where json file will be saved/read", default=os.getenv('JSON_PATH'))
    parser.add_argument("--data", help="Path to the dataset",default=os.getenv('DATA_PATH'))
    parser.add_argument("--save", help="Path to where checkpoints will be saved",default=os.getenv('SAVE_PATH'))
    parser.add_argument("--insize", help="Size of input image",default=32)
    parser.add_argument("--outsize", help="Size of output image",default=32)
    parser.add_argument("--batch", help="Size of batches",default=32)
    parser.add_argument("--epochs", help="Number of epochs",default=300)

    parser.add_argument("--video", help="Number of batches between every visualization",default=10)

    parser.add_argument('--no-video', help = "Set for no visual debug", dest='video', action='store_false')

    parser.add_argument("--load-chkp", help="Path to checkpoints to be loaded",dest='loadchkp',default="")

import os
import argparse
from dotenv import load_dotenv
def parse_argv():
    load_dotenv()
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", help="Path where json file will be saved/read", default=os.getenv('JSON_PATH'))
    parser.add_argument("--data", help="Path to the dataset",default=os.getenv('DATA_PATH'))
    parser.add_argument("--save", help="Path to where checkpoints will be saved",default=os.getenv('SAVE_PATH'))
    parser.add_argument("--insize", help="Size of input image",default=32)
    parser.add_argument("--outsize", help="Size of output image",default=32)
    parser.add_argument("--batch", help="Size of batches",default=32)
    parser.add_argument("--epochs", help="Number of epochs",default=300)

    parser.add_argument("--video", help="Number of batches between every visualization",default=10)

    parser.add_argument('--no-video', help = "Set for no visual debug", dest='video', action='store_false')

    parser.add_argument("--load-chkp", help="Path to checkpoints to be loaded",dest='loadchkp',default="")
    
    parser.set_defaults(colab=False)
    parser.add_argument('--colab', help = "Set for no visual debug", action='store_true')
 

    ar=parser.parse_args()

    return (ar.json,
            ar.data,
            ar.save,
            (int(ar.insize), int(ar.insize)),
            (ar.outsize, ar.outsize),
            int(ar.batch),
            int(ar.epochs),
            ar.video,
            ar.loadchkp,
            ar.colab)