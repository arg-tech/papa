import json
import re

'''
! This script assumes the presence of a 'text' field in the XAIF !
'''

# Convert if necessary

def ova2_to_ova3(xaif_in, verbose=False):
    # If not OVA2, return as is
    if 'AIF' in xaif_in:
        if verbose:
            print('already OVA3')
        if 'text' not in xaif_in:
            print('no text field!')
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


###########
# Utility #
###########

# Given ID of an L-node, return whether it is reported speech or not
def is_reported_speech(node_id, all_nodes):
    if all_nodes[node_id]['type'] != 'L':
        return False
    else:
        # Get incoming, and check if any is a YA other than Analysing
        for node_in in all_nodes[node_id]['ein']:
            if all_nodes[node_in]['type'] == 'YA' and all_nodes[node_in]['text'] != 'Analysing':
                return True
        
        # Have gotten through all incoming nodes without finding a non-Analysing YA incoming
        return False


# Return speaker of L-node
def l_node_speaker(node_id, all_nodes):
    # ya_anchor is now the furthest back L-node in the chain: return the speaker
    splits = all_nodes[node_id]['text'].split(':')
    if len(splits) < 2:
        spkr = ''
        print(f"L-node with no recognisable speaker:\t{all_nodes[node_id]['nodeID']}")
    else:
        spkr = splits[0].strip()
    return spkr


# Given a reported speech L-node, return speaker of the reporting L-node
def reporting_speaker(l_node_id, all_nodes):
    quoting_ya = [n for n in all_nodes[l_node_id]['ein'] if all_nodes[n]['type'] == 'YA']

    # No incoming YA to this node, or the YA is Analysing: this was the original locution so should return this spkr
    if len(quoting_ya) == 0 or 'Analysing' in [all_nodes[q]['text'] for q in quoting_ya]:
        return l_node_speaker(l_node_id, all_nodes)

    # Incoming YA: this is reported, so check the L-node anchoring that YA
    else:
        if len(quoting_ya) > 1:
            print(f"Multiple incoming YAs to L-node {l_node_id}: {all_nodes[l_node_id]['text']}")
            print(f"Trying first only")
        
        ya_anchor = [n for n in all_nodes[quoting_ya[0]]['ein'] if all_nodes[n]['type'] == 'L']

        # Report if anything unexpected found (too many/few L-nodes)
        if len(ya_anchor) < 1:
            print(f"Can't find L-node for YA {quoting_ya[0]}")
            return ''
        if len(ya_anchor) > 1:
            print(f"Multiply-anchored YA {quoting_ya}: anchored by ", *ya_anchor)
        
        return reporting_speaker(ya_anchor[0], all_nodes)


# Given an l_node, return ID of nonreported L-node
# (if non-reported speech, self; if reported speech, L-node introudcing it)
def start_of_l_chain(l_node_id, all_nodes):
    quoting_ya = [n for n in all_nodes[l_node_id]['ein'] if all_nodes[n]['type'] == 'YA']
    if len(quoting_ya) == 0 or 'Analysing' in [all_nodes[q]['text'] for q in quoting_ya]:
        return l_node_id
    else:
        if len(quoting_ya) > 1:
            print(f"Multiple incoming YAs to L-node {l_node_id}: {all_nodes[l_node_id]['text']}")
            print(f"Trying first only")
        
        ya_anchor = [n for n in all_nodes[quoting_ya[0]]['ein'] if all_nodes[n]['type'] == 'L']

        # Report if anything unexpected found (too many/few L-nodes)
        if len(ya_anchor) < 1:
            print(f"Can't find L-node for YA {quoting_ya[0]}")
            return ''
        if len(ya_anchor) > 1:
            print(f"Multiply-anchored YA {quoting_ya}: anchored by ", *ya_anchor)
        
        return start_of_l_chain(ya_anchor[0], all_nodes)



###############################
# Pre-Analytic Info Gathering #
###############################

# Take a list of OVA nodes and return the IDs of those set to non-visible
def invisible_nodes(xaif: dict) -> list:
    metanode_ids = [n['nodeID'] for n in xaif['OVA']['nodes'] if not n['visible']]
    return metanode_ids


# Cut analysing-related nodes and edges from the AIF
def remove_all_meta(xaif: dict, verbose=False) -> dict:
    if verbose:
        print("*** Removing extra nodes and edges ***")

    # Has OVA section: remove whatever's invisble
    if 'OVA' in xaif:
        if verbose:
            print('Removing nodes marked as invisible in OVA')
        # Get the IDs for the meta nodes (nodes set to non-visible in OVA)
        metanode_ids = invisible_nodes(xaif)
        if verbose:
            print(f"Found {len(metanode_ids)} invisible nodes: ", metanode_ids)

    # File has no OVA section: find meta nodes manually
    else:
        # return xaif
        if verbose:
            print("No OVA section: removing based on 'Analysing' nodes")
        
        analysing_node_ids = [n['nodeID'] for n in xaif['AIF']['nodes'] if n['text'] == 'Analysing' and n['type'] == 'YA']
        
        # print("Node list:")
        # for n in xaif['AIF']['nodes']:
        #     # if n['text'] == 'Analysing' and n['type'] == 'YA':
        #     print('\t',n)
       
        # print("Edge list:")
        # for e in xaif['AIF']['edges']:
        #     # if e['toID'] in analysing_node_ids:
        #     print('\t', e) 
        ids_to_analysing = [e['fromID'] for e in xaif['AIF']['edges'] if e['toID'] in analysing_node_ids]
        metanode_ids = analysing_node_ids + ids_to_analysing
        if verbose:
            print(metanode_ids)

    nonmeta_nodes = [n for n in xaif['AIF']['nodes'] if n['nodeID'] not in metanode_ids]

    # Remove the edges involving meta nodes from the list of AIF nodes.
    # Make an alternative 'edges' list for 'AIF' with the edges connected to a meta node stripped out
    # (Remove any edge from a meta node: visible nodes don't have edges to invisible nodes, unidirectional
    nonmeta_edges = [e for e in xaif['AIF']['edges'] if e['fromID'] not in metanode_ids]

    if verbose:
        print()
        print(f'Nodes to cut ({len(metanode_ids)}):', metanode_ids)
        print(f'Nodes to keep ({len(nonmeta_nodes)}):', [n['nodeID'] for n in nonmeta_nodes])
        print(f'Edges to cut (?)')
        print(f'Edges to keep ({len(nonmeta_edges)}):', nonmeta_edges)

    if verbose:
        print("\nOld counts:")
        print("\tNodes: ", len(xaif['AIF']['nodes']))
        print("\tEdges: ", len(xaif['AIF']['edges']))

    # Replace 'nodes', 'edges' and 'schemefulfillments' in AIF with the stripped versions
    xaif['AIF']['nodes'] = nonmeta_nodes
    xaif['AIF']['edges'] = nonmeta_edges

    if verbose:
        print("\nNew counts:")
        print("\tNodes: ", len(xaif['AIF']['nodes']))
        print("\tEdges: ", len(xaif['AIF']['edges']))


    return xaif


# Create dicts of nodes indexed by nodeID, with entry for each node
def node_setup(xaif):
    all_nodes = {}
    for n in xaif['AIF']['nodes']:
        all_nodes[n['nodeID']] = {}
        
        all_nodes[n['nodeID']]['nodeID'] = n['nodeID']
        all_nodes[n['nodeID']]['type'] = n['type']
        all_nodes[n['nodeID']]['text'] = n['text']
        all_nodes[n['nodeID']]['speaker'] = []
        all_nodes[n['nodeID']]['ein'] = []
        all_nodes[n['nodeID']]['eout'] = []
        all_nodes[n['nodeID']]['agree'] = []
        # all_nodes[n['id']]['support'] = 0
        all_nodes[n['nodeID']]['saidby'] = []
        all_nodes[n['nodeID']]['chron'] = -1 # for ordering locs
        all_nodes[n['nodeID']]['introby'] = [] # for associating props with initially-introducing locs
    
    return all_nodes


def add_edge_info(xaif, all_nodes):
    for e in xaif['AIF']['edges']:
        if e['fromID'] in all_nodes and e['toID'] in all_nodes:
            all_nodes[e['toID']]['ein'].append(e['fromID'])
            all_nodes[e['fromID']]['eout'].append(e['toID'])
    # cut erroneous duplicates
    for n in all_nodes:
        all_nodes[n]['ein'] = list(set(all_nodes[n]['ein']))
        all_nodes[n]['eout'] = list(set(all_nodes[n]['eout']))
    return all_nodes


def path_to_start(l_node, all_nodes):
    # Nodes with edges to L-node other than TA-nodes
    prev_nodes = [n for n in all_nodes[l_node]['ein'] if all_nodes[n]['type'] != 'TA']
    for n in prev_nodes:
        if all_nodes[n]['type'] not in ['I', 'RA', 'CA', 'MA']:
                return path_to_start(n, all_nodes)
    # if prev list is empty, this was start: return it
    return l_node


# also adds locution associations to i-nodes
def add_speakers(all_nodes, verbose=False):
    said = {}

    for n in [n for n in all_nodes if all_nodes[n]['type'] == 'L']:

        # Get L speakers
        splits = all_nodes[n]['text'].split(':')
        if len(splits) < 2:
            spkr = ''
            print(f"L-node with no recognisable speaker:\t{all_nodes[n]['nodeID']}")
        else:
            spkr = splits[0].strip()
            
            # Record node-wise
            all_nodes[n]['speaker'] = [spkr]
            

            if not is_reported_speech(n, all_nodes):            
                # Record speaker-wise
                if spkr in said:
                    said[spkr].append(all_nodes[n]['nodeID'])
                else:
                    said[spkr] = [all_nodes[n]['nodeID']]

        # Add for associated I-nodes
        if not is_reported_speech(n, all_nodes):
            for e_out in all_nodes[n]['eout']:
                if all_nodes[e_out]['type'] == 'YA':
                    ya = e_out
                    for ya_out in all_nodes[ya]['eout']:
                        if all_nodes[ya_out]['type'] == 'I':
                            if spkr != '':
                                all_nodes[ya_out]['saidby'].append(spkr)
                                said[spkr].append(all_nodes[ya_out]['nodeID'])

                            all_nodes[ya_out]['introby'].append(all_nodes[n]['nodeID'])
        # Reported speech: I-node should be attributed to the quoting speaker
                            
            
        else:
            # Get quoting speaker
            quoter = reporting_speaker(n, all_nodes)
            source = path_to_start(n, all_nodes)
            # Add an introby to self

            all_nodes[n]['introby'].append(source)

            # Add to I-node if directly connected
            for e_out in all_nodes[n]['eout']:
                if all_nodes[e_out]['type'] == 'YA':
                    ya = e_out
                    for ya_out in all_nodes[ya]['eout']:
                        if all_nodes[ya_out]['type'] == 'I' and quoter != '':
                            # record node-wise only
                            all_nodes[ya_out]['saidby'].append(quoter)
                            
                            all_nodes[ya_out]['introby'].append(source)
                            # all_nodes[ya_out]['introby'].append(all_nodes[n]['nodeID'])

    # Meeds cleaning/merging with following chunk, but leaving as is for now
    # Get TA speakers
    for n in [n for n in all_nodes if all_nodes[n]['type'] == 'TA']:
        for e_in in all_nodes[n]['ein']:
            for e_out in all_nodes[n]['eout']:
                if (all_nodes[e_in]['type'] == 'L' and all_nodes[e_out]['type'] == 'L'):
                    l1 = e_in
                    l2 = e_out

                    splits = all_nodes[l2]['text'].split(':')
                    if len(splits) < 2:
                        print(f"L-node with no recognisable speaker:\t{all_nodes[n]['nodeID']}")
                        spkr = ''
                    else:
                        spkr = splits[0].strip()
                    
                    for ta_out in all_nodes[n]['eout']:
                        if all_nodes[ta_out]['type'] == 'YA':
                            for i_out in all_nodes[ta_out]['eout']:
                                if all_nodes[i_out]['type'] == 'I' and spkr != '':
                                    # Record node-wise
                                    all_nodes[i_out]['saidby'].append(spkr)
                                    all_nodes[i_out]['introby'].append(l2)

                                    # Record speaker-wise
                                    if spkr in said:
                                        said[spkr].append(all_nodes[i_out]['nodeID'])
                                    else:
                                        said[spkr] = all_nodes[i_out]['nodeID']
                                        

        # Adding speaker attribution to arg relations based on speaker of L-nodes descended from the anchoring TA
        # Requires update: original doesn't trace back in case of reported speech
        # for ta_out in all_nodes[n]['eout']:
        #     # speakers of L-nodes descended from this TA
        #     # l_spkrs = [all_nodes[l]['speaker'] for l in all_nodes[n]['eout'] if all_nodes[l]['type'] == 'L']
        #     l_spkrs = [s for l in all_nodes[n]['eout'] if all_nodes[l]['type'] == 'L' for s in all_nodes[l]['speaker']]
        #     if all_nodes[ta_out]['type'] == 'YA':
        #         for ya_out in all_nodes[ta_out]['eout']:
        #             if all_nodes[ya_out]['type'] == 'RA' or 'CA' or 'MA':
        #                 all_nodes[ya_out]['speaker'] = l_spkrs


    # Assign TA speakers
    for n in [n for n in all_nodes if all_nodes[n]['type'] == 'TA']:
        # (If well-formed this L list should have exactly one entry)
        l_out = [l for l in all_nodes[n]['eout'] if all_nodes[l]['type'] == 'L']
        spkrs = []
        for l in l_out:
            spkrs = spkrs + all_nodes[l]['speaker']
        all_nodes[n]['speaker'] = list(set(spkrs))

    # Get YA speakers: the speaker of any L or TA that anchors the YA
    for n in [n for n in all_nodes if all_nodes[n]['type'] == 'YA']:
        if verbose:
            print(f"\nAttributing {all_nodes[n]['text']} YA {n}")
        # Check each incoming node to the YA that is a TA or L
        for e in [e for e in all_nodes[n]['ein'] if all_nodes[e]['type'] in ['L', 'TA']]:
            all_nodes[n]['speaker'] = all_nodes[n]['speaker'] + all_nodes[e]['speaker']
        all_nodes[n]['speaker'] = list(set(all_nodes[n]['speaker']))

        
    # Add speaker attribution to arg relations on basis of anchoring YA
    # Requires update: doesn't trace back in case of reported speech
    for n in [n for n in all_nodes if all_nodes[n]['type'] in ['RA', 'CA', 'MA']]:

        ya_in = [ya for ya in all_nodes[n]['ein'] if all_nodes[ya]['type'] == 'YA']
        if verbose:
            print(f"\nChecking {all_nodes[n]['type']}-node {n}")
            print(f"Anchored by YA(s): ", ya_in)
            for y in ya_in:
                print(f"\t{y}\t{all_nodes[y]['text']}")

        if len(ya_in) == 0:
            print(f"Unanchored {all_nodes[n]['type']}-node: {n}")
        elif len(ya_in) == 1:
            all_nodes[n]['speaker'] = all_nodes[ya_in[0]]['speaker']
            if verbose:
                print(f"List ya_in:", ya_in)
                print(all_nodes[ya_in[0]])

                print(f"\tJust one YA, attributed as:", all_nodes[ya_in[0]]['speaker'])
                print(f"\tAttributing {n} to ", all_nodes[n]['speaker'])
        else:
            all_spkrs = []
            for ya in ya_in:
                all_spkrs = all_spkrs + all_nodes[ya]['speaker']
            all_nodes[n]['speaker'] = list(set(all_spkrs))
            
    
    return all_nodes, said

# Provided that add_speakers has been performed, add speakers to YAs and TAs based on
# TA -> speaker(s) of the descending L-node(s)
# YA -> speaker(s) of the anchoring L or TA node
def add_assumed_speakers(all_nodes):
    for ta in [n for n in all_nodes if all_nodes[n]['type'] == 'TA']:
        l_out = [n for n in all_nodes[ta]['eout'] if all_nodes[n]['type'] == 'L']
        for l in l_out:
            all_nodes[ta]['speaker'] = all_nodes[ta]['speaker'] + all_nodes[l]['speaker']

    for ya in [n for n in all_nodes if all_nodes[n]['type'] == 'YA']:
        ya_anchor = [n for n in all_nodes[ya]['ein'] if all_nodes[n]['type'] == 'L' or all_nodes[n]['type'] == 'TA']
        for anchor in ya_anchor:
            all_nodes[ya]['speaker'] = all_nodes[ya]['speaker'] + all_nodes[anchor]['speaker']
    
    return all_nodes


def add_agreement(all_nodes):
    for n in all_nodes:
        if all_nodes[n]['type'] == 'YA' and all_nodes[n]['text'] == ('Agreeing' or 'Asserting'):
            for e_in in all_nodes[n]['ein']:
                for e_out in all_nodes[n]['eout']:
                    if all_nodes[e_in]['type'] == 'L' and all_nodes[e_out]['type'] == 'I':
                        all_nodes[e_out]['agree'] = all_nodes[e_out]['agree'] + all_nodes[e_in]['speaker']
                        # all_nodes[e_out]['agree'].append(all_nodes[e_in]['speaker'])
        
        if all_nodes[n]['type'] == 'RA':
            for e_in in all_nodes[n]['ein']:
                for e_out in all_nodes[n]['eout']:
                    all_nodes[e_out]['agree'] = list(set(all_nodes[e_out]['agree'] + all_nodes[e_in]['saidby']))
        
        if all_nodes[n]['type'] == 'MA':
            for e_in in all_nodes[n]['ein']:
                for e_out in all_nodes[n]['eout']:
                    all_nodes[e_out]['agree'] = list(set(all_nodes[e_out]['agree'] + all_nodes[e_in]['agree']))
    return all_nodes


# Order on basis of text
def locution_markup_sort(old_xaif_text): 
    marked_loc_spans = re.findall('(?<=id="node)\d+_\d+', old_xaif_text)
    return marked_loc_spans

def add_loc_order(xaif, all_nodes, verbose=False):
    try:
        loc_order = locution_markup_sort(xaif['text'])
        counter = 1

        if verbose:
            print(loc_order)
            print(all_nodes.keys())

        for num in loc_order:
            try: 
                all_nodes[num]['chron'] = counter
                counter += 1
            except KeyError:
                print(f"Node {num} found in text but no node {num} found in node list")
    except:
        print("No text field found so cannot place locutions in correct order")
    return all_nodes


# Add nodeID of earliest locution which anchors each proposition (and reported speech)
def add_prop_earliest_intro(all_nodes):
    for n in all_nodes:
        if n['type'] == 'I':
            root_l_nodes = [all_nodes[root] for root in n['introby'] if all_nodes[root]['chron'] != -1]
            earliest_root = sorted(root_l_nodes, key=lambda x: x['chron'])[0]
            all_nodes[n['nodeID']]['introby'] = earliest_root['nodeID']
    return all_nodes



def spkr_wordcounts(xaif, verbose=False):
    text = xaif['text']
    # Remove char codes
    text = re.sub("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", '', text)

    text = text.replace('\n', '')

    # Assumes bracketed timestamps
    # Added comma
    split_list = re.split("([\.\"?'!’…,]| -+)(?=[\w\s\d-]+\[)", text)
    if verbose:
        print("Split texts:")
        for s in split_list:
            print(f"\t{s}")
        print()
    turn_list = []
    if verbose:
        print("Turns:")
    
    for i, item in enumerate(split_list):
        if i%2 == 0:
            turn_list.append(item)
        else:
            turn_list[-1] = turn_list[-1] + item
        # if verbose:
        #     print(turn_list[-1])
    

    spkr_wordcounts = {}
    
    for t in turn_list:
        turn_split = re.split('\[.*\]', t)
        if verbose:
            print("Turn splits as:")
            for i, t in enumerate(turn_split):
                print(f"\t{i}: {t.strip()}")
        spkr = turn_split[0].strip()
        try:
            content = turn_split[1].strip()
        except IndexError:
            print("\tTURN CONTENT NOT FOUND: ", turn_split)
            content = ''
        # print(f"Speaker: {spkr}")
        # print(f"Content: {content}\n")

        if spkr not in spkr_wordcounts.keys():
            spkr_wordcounts[spkr] = {'wordcount': 0}
            # spkr_wordcounts[spkr] = len(content.split())
        # else:
        spkr_wordcounts[spkr]['wordcount'] += len(content.split())
    
    return spkr_wordcounts


def xaif_preanalytic_info_collection(xaif, verbose=False):
    
    # Cut meta content first
    xaif = ova2_to_ova3(xaif)
    xaif = remove_all_meta(xaif)

    all_nodes = node_setup(xaif)
    all_nodes = add_edge_info(xaif, all_nodes)

    all_nodes, said = add_speakers(all_nodes, verbose=verbose)
    # all_nodes = add_assumed_speakers(all_nodes)
    all_nodes = add_agreement(all_nodes)

    all_nodes = add_loc_order(xaif, all_nodes)


    return all_nodes, said