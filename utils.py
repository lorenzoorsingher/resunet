import os
import argparse
import json
from dotenv import load_dotenv

def parse_argv_train():
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
            (int(ar.outsize), int(ar.outsize)),
            int(ar.batch),
            int(ar.epochs),
            int(ar.video),
            ar.loadchkp,
            ar.colab)

def parse_argv_inf():
    load_dotenv()
    parser=argparse.ArgumentParser()

    parser.add_argument("--input", help="Path to the image", default="")
    parser.add_argument("--json", help="Path where json file will be saved/read", default=os.getenv('JSON_PATH'))
    parser.add_argument("--save", help="Path to where checkpoints will be saved",default=os.getenv('SAVE_PATH'))
    parser.add_argument("--size", help="Size of input image",default=32)

    parser.add_argument("--load-chkp", help="Path to checkpoints to be loaded",dest='loadchkp',default="")
    
    parser.set_defaults(colab=False)
    parser.add_argument('--colab', help = "Set for no visual debug", action='store_true')
 

    ar=parser.parse_args()

    return (ar.input,
            ar.json,
            (int(ar.size), int(ar.size)),
            ar.loadchkp,
            ar.colab)

def save_loss(dict, path):
    with open(path + "loss.json", 'w') as fp:
        json.dump(dict, fp)