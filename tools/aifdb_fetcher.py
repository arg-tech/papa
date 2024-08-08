# Download files associated with argument maps from AIFdb -- ova map versions
# April/May 2023

import os
import json
import wget
import urllib.request
import sys
import argparse


def get_maps_orig(corpora: list, map_type='aif'):
    if not corpora:
        # # Naming convention for this collection is [EN if English][task][group]
        # corporalist = [
        #     'moonlandingde1',
        #     'enjamestownde1',
        #     'moonlandingde2',
        #     'enjamestownde2',
        #     'moonlandingpl',
        #     'enjamestownpl',
        #     'jamestownit',
        #     'enmoonlandingit'
        # ]
        print("!! Need a corpus (or corpora) to fetch. Please include at least one corpus shortname.")
        sys.exit()
    else:
        corporalist = corpora

    # Nodeset IDs for each map in a corpus can be found at
    # http://corpora.aifdb.org/nodesets.php?shortname=[shortname here]
    # This comes in the form: {"nodeSets": [00000, ...]}

    if map_type not in ['aif', 'ova']:
        print(f"Can't get files for map type '{map_type}'. Please try again with 'aif' or 'ova'.")
        sys.exit()
    

    for corpus in corporalist:
        print(f"Getting {map_type.upper()} files for corpus {corpus}")

        ####################
        # LOCATION TO SAVE #
        ####################

        # Make a named corpus directory to put files in if not existing
        # cwd = os.getcwd()
        # print(f"Current working directory: {cwd}")
        # aif_folder = "/Users/eimear/Documents/Local AIFdb"
        aif_folder = "/Users/eimear/Documents/Paper_writing/ArgDiffs/data"
        # folderpath = os.path.join(cwd, corpus, map_type) # cwd+f'/{corpus}'
        folderpath = os.path.join(aif_folder, corpus) # cwd+f'/{corpus}'
        if not os.path.exists(folderpath):
            print(f"Making new folder: {folderpath}")
            os.makedirs(folderpath)
        else:
            print(f"Destination: {folderpath}")
        print()


        ################
        # FILES TO GET #
        ################

        # Get the list of nodesets in the corpus
        url = f"http://corpora.aifdb.org/nodesets.php?shortname={corpus}"
        print(f"Getting the nodeset at {url}")
        print()

        # Get the list of nodesets in the corpus
        # Opens as a one-entry python dictionary with key nodeSets, and value as list of node numbers (int)
        # This is a copypaste from QTTransferOVA.py
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            # print("List of nodes: ", data['nodeSets'])


        #############
        # GET FILES #
        #############

        missing = []
        # if map_type == 'ova':
            # url_base = "http://ova.arg.tech/db/"
        
        
        # Go through the nodes, downloading nodeset at each one if not already downloaded
        for i, n in enumerate(data['nodeSets']):
            print("----")
            print(f"Checking for argmap at nodeset {n} (node {i+1}/{len(data['nodeSets'])})")
            filepath = os.path.join(folderpath, f"{str(n)}.json")

            print(f"Checking for file {filepath}")
            if os.path.exists(filepath):
                print("Already downloaded!")
            else:
                print("File doesn't exist yet.")
                # fileurl = url_base + n
                if map_type == 'ova':
                    fileurl = f"http://ova.arg.tech/db/{n}"
                    # print(f"\nGetting OVA-friendly file", end='')
                else:
                    fileurl = f"http://www.aifdb.org/json/{n}"
                    # print(f"\nGetting json file at {fileurl}",  end='')

                print(f" Getting url {fileurl}... ")
                try:
                    wget.download(fileurl, filepath)
                    print("\nDone")
                except urllib.error.HTTPError:
                    missing.append(n)
                    print(f"\n404: Nodeset {n} appears not to exist")
        print("----")
        print(f"Finished with the argument maps in corpus {corpus}")
        print("==========")

    print("Finished all corpora in the list")
    if len(missing) > 0:
        print(f"{len(missing)}/{len(data['nodeSets'])} nodesets in the list were misisng.")
        print("The following nodesets were not found: ", *missing)



def get_maps(corpora, dir_out, map_type):
    if not corpora:
        print("!! Need a corpus (or corpora) to fetch. Please include at least one corpus shortname.")
        sys.exit()
    else:
        corporalist = corpora

    # Nodeset IDs for each map in a corpus can be found at
    # http://corpora.aifdb.org/nodesets.php?shortname=[shortname here]
    # This comes in the form: {"nodeSets": [00000, ...]}

    if map_type not in ['aif', 'ova']:
        print(f"Can't get files for map type '{map_type}'. Please try again with 'aif' or 'ova'.")
        sys.exit()
    

    for corpus in corporalist:
        print(f"Getting {map_type.upper()} files for corpus {corpus}")

        ####################
        # LOCATION TO SAVE #
        ####################

        # Make a named corpus directory to put files in if not existing
        # cwd = os.getcwd()
        # print(f"Current working directory: {cwd}")
        # aif_folder = "/Users/eimear/Documents/Local AIFdb"
        # aif_folder = "/Users/eimear/Documents/Paper_writing/ArgDiffs/data"
        # folderpath = os.path.join(cwd, corpus, map_type) # cwd+f'/{corpus}'
        # folderpath = os.path.join(aif_folder, dir_out) # cwd+f'/{corpus}'
        if not os.path.exists(dir_out):
            print(f"Making new folder: {dir_out}")
            os.makedirs(dir_out)
        else:
            print(f"Destination: {dir_out}")
        print()


        ################
        # FILES TO GET #
        ################

        # Get the list of nodesets in the corpus
        url = f"http://corpora.aifdb.org/nodesets.php?shortname={corpus}"
        print(f"Getting the nodeset at {url}")
        print()

        # Get the list of nodesets in the corpus
        # Opens as a one-entry python dictionary with key nodeSets, and value as list of node numbers (int)
        # This is a copypaste from QTTransferOVA.py
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            # print("List of nodes: ", data['nodeSets'])


        #############
        # GET FILES #
        #############

        missing = []
        # if map_type == 'ova':
            # url_base = "http://ova.arg.tech/db/"
        
        
        # Go through the nodes, downloading nodeset at each one if not already downloaded
        for i, n in enumerate(data['nodeSets']):
            print("----")
            print(f"Checking for argmap at nodeset {n} (node {i+1}/{len(data['nodeSets'])})")
            filepath = os.path.join(dir_out, f"{str(n)}.json")

            print(f"Checking for file {filepath}")
            if os.path.exists(filepath):
                print("Already downloaded!")
            else:
                print("File doesn't exist yet.")
                # fileurl = url_base + n
                if map_type == 'ova':
                    fileurl = f"http://ova.arg.tech/db/{n}"
                    # print(f"\nGetting OVA-friendly file", end='')
                else:
                    fileurl = f"http://www.aifdb.org/json/{n}"
                    # print(f"\nGetting json file at {fileurl}",  end='')

                print(f" Getting url {fileurl}... ")
                try:
                    wget.download(fileurl, filepath)
                    print("\nDone")
                except urllib.error.HTTPError:
                    missing.append(n)
                    print(f"\n404: Nodeset {n} appears not to exist")
        print("----")
        print(f"Finished with the argument maps in corpus {corpus}")
        print("==========")

    print("Finished all corpora in the list")
    if len(missing) > 0:
        print(f"{len(missing)}/{len(data['nodeSets'])} nodesets in the list were misisng.")
        print("The following nodesets were not found: ", *missing)

######################################
# Run from command line, if you want #
######################################

if __name__ == "__main__":
    # corpora = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="aif or ova", default='aif')
    parser.add_argument("--corpora", '-c', nargs='*', default = "")
    parser.add_argument("--directory", '-d', nargs='*', default = "")
    args = parser.parse_args()
    get_maps(args.corpora, args.type, args.directory)
