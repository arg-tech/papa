'''
Functions for handling OVA files
- Update node IDs in OVA2 files to be unique
- Convert OVA2 files to OVA3 files
- Combine all OVA3 files in a directory into a single .json.
'''

import re
import json
from glob import glob
import os
import shutil


# Resave all .json files in a (possibly new) directory, adding unique identifiers to the nodes in any OVA2 files
def ova2_unique_ids(dir_in, dir_out):
    files_in = glob(f"{dir_in}/*.json")
    for old_file in files_in:
        # Get map
        with open(old_file) as f:
            argmap = json.loads(f.read())

        mapname = os.path.basename(old_file).split('.')[0]

        # New ova file: just copy
        if 'AIF' in list(argmap.keys()):
            with open(f"{dir_out}/{mapname}.json", 'w') as file:
                json.dump(argmap, file, indent=4)
        
        # Old ova file: update node names in nodes, edges and text
        else:
            for n in argmap['nodes']:
                n['id'] = str(n['id']) +  f"_{mapname}"

            for e in argmap['edges']:
                e['to']['id'] = str(e['to']['id']) +  f"_{mapname}"
                e['from']['id'] = str(e['from']['id']) +  f"_{mapname}"

            argmap['analysis']['txt'] = re.sub(r'(id=\"node)(\d+)(\")', r'\1\2_' + mapname + r'\3', argmap['analysis']['txt'])

            with open(f"{dir_out}/{mapname}.json", 'w') as file:
                json.dump(argmap, file, indent=4)


# Accept an OVA2 xaif dict and return it in OVA3 format
def ova2_to_ova3(xaif_in):
    # If not OVA2, return as is
    if 'AIF' in xaif_in:
        print('already OVA3')
        return xaif_in
    
    xaif_out = {
        "AIF": {
            "nodes": [],
            "edges": [],
            "schemefulfillments": [],
            "participants": [],
            "locutions": [],
            "descriptorfulfillments": [],
            "cqdescriptorfulfillments": []
        },
        "text": '',
        "OVA": { # Could leave this empty tbh: will be unworkable in OVA anyway.
            "firstname": 'Convertor',
            "surname": 'Script',
            "url": '',
            "nodes": [],
            "edges":[]
        }  
    }

    # Nodes
    for n in xaif_in['nodes']:
        if 'timestamp' not in n:
            n['timestamp'] = ''

        xaif_out['AIF']['nodes'].append({
            'nodeID': n['id'],
            'text': n['text'],
            'type': n['type']
        })

        xaif_out['AIF']['schemefulfillments'].append({
            'nodeID': n['id'],
            'schemeID': n['scheme']
        })

        if n['type'] == 'L':
            xaif_out['AIF']['locutions'].append({
                'nodeID': n['id'],
                'personID': n['participantID'],
                'start': None,
                'end': None
            })
        
        xaif_out['OVA']['nodes'].append({
            'nodeID': n['id'],
            'visible': n['visible'],
            'x': n['x'],
            'y': n['y'],
            'timestamp': n['timestamp']
        })

    # Edges
    edge_counter = 1
    for e in xaif_in['edges']:
        xaif_out['AIF']['edges'].append({
            'edgeID': edge_counter,
            'fromID': e['from']['id'],
            'toID': e['to']['id']
        })

        xaif_out['OVA']['edges'].append({
            'fromID': e['from']['id'],
            'toID': e['to']['id'],
            'visible': e['visible']
        })

        edge_counter += 1

    # Participants
    for p in xaif_in['participants']:
        xaif_out['AIF']['participants'].append({
            'participantID': p['id'],
            'firstname': p['firstname'],
            'surname': p['surname']
        })

    xaif_out['text'] = xaif_in['analysis']['txt']

    return xaif_out



# Oh shit. Are there any episodes with a MIX of OVA types? In this case yeah, need to be merging AIF not OVA...
# But... let's just have the option to merge it yourself in the meantime, it won't take too long.

# Combine all json files in a directory, assuming they're OVA3 files
def combine_ova3(dir_in, file_out, verbose=False):
    json_list = glob(f"{dir_in}/*.json")
    # json_list = glob(f"{dir_in}/*_ova3.json")
    
    if verbose:
        print("Combining files in the directory ", dir_in)
        print(f"{len(json_list)} files found.")
    
    file_txts = []
    
    combined_xaif = {
        "AIF": {
            "nodes": [],
            "edges": [],
            "schemefulfillments": [],
            "participants": [],
            "locutions": [],
            "descriptorfulfillments": [],
            "cqdescriptorfulfillments": []
        },
        "text": '',
        "OVA": { # Could leave this empty tbh: will be unworkable in OVA anyway.
            "firstname": 'Combination',
            "surname": 'Script',
            "url": '',
            "nodes": [],
            "edges":[]
        }  
    }
    

    # Assemble the AIF (atemporal)
    for j in json_list:
        with open(j, 'r') as file:
            xaif_in = json.loads(file.read())

        # Get OVA3 version if it's OVA2
        if 'AIF' not in xaif_in:
            xaif_in = ova2_to_ova3(xaif_in)
            
        for n in xaif_in['AIF']['nodes']:
            if n not in combined_xaif['AIF']['nodes']:
                combined_xaif['AIF']['nodes'].append(n)
        
        for e in xaif_in['AIF']['edges']:
            if e not in combined_xaif['AIF']['edges']:
                combined_xaif['AIF']['edges'].append(e)
        
        for l in xaif_in['AIF']['locutions']:
            if l not in combined_xaif['AIF']['locutions']:
                combined_xaif['AIF']['locutions'].append(l)

        file_txts.append(xaif_in["text"])
    

    # Order and concatenate the texts
    stamp_list = []
    for i, t in enumerate(file_txts):
        if t == "Enter your text here...":
            if verbose:
                print(f"Text {i} is 'Enter your text here...'")
            continue
        try:
            first_stamp = re.search('\[[\d]+:\d\d:\d\d\]', t).group(0)
            if not re.search('\[\d\d:', first_stamp):
                # print("not a double-digit start")
                first_stamp = re.sub('\[', '[0', first_stamp)
            # print(first_stamp)
            # Check for number of digits:
            stamp_list.append((first_stamp, i))
        except AttributeError:
            print(f"No timestamp found in text {i}: {t}")
    if verbose:
        print("Stamp list: ", stamp_list)
    stamp_list.sort(key=lambda x: x[0])
    if verbose:
        print("Sorted stamp list: ", stamp_list)

    # ordered_texts = []
    # if verbose:
        # print("Ordered texts:")
    for i, s in enumerate(stamp_list):
        # ordered_texts.append(file_txts[s[1]])
        combined_xaif['text'] += ' ' + file_txts[s[1]]
        # if verbose:
            # print(f'\tTEXT {i}:', file_txts[s[1]])
    

    # Write the final result
    if verbose:
        print("Writing to ", file_out)
    with open(file_out, 'w') as file:
        json.dump(combined_xaif, file, indent=4)


# Does both steps of OVA2 -> OVA3
# 1) unique IDs
# 2) format conversion
def ova_all_ova3(dir_in, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    # Make IDs unique
    ova2_unique_ids(dir_in, dir_out)

    # Resave in OVA3 format
    file_names = glob(f"{dir_out}/*.json")
    for file in file_names:
        print(file)
        
        with open(file, 'r') as f:
            orig_xaif = json.loads(f.read())
        
        converted = ova2_to_ova3(orig_xaif)

        with open(file, 'w') as f:
            json.dump(converted, f, indent=4)
            