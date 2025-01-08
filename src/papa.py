import json
from . import xaif_toolbox as ova3
from . import analytics
import statistics
import sys
import os
import regex as re
import pandas as pd

def all_analytics(xaif, node_level=False, speaker=False, forecast=False):
    xaif = ova3.ova2_to_ova3(xaif)

    # This involves redundancy currently bc all analytics are designed to call the info collection
    # so that they can be run individually with just the XAIF...


    # Speaker-level analytics
    if speaker:
        if 'AIF' in xaif.keys():
            wordcounts = ova3.spkr_wordcounts(xaif)
            spkr_analytic_list = []
                
        spkr_rel_counts = analytics.arg_relation_counts(xaif, speaker=True)

        spkr_analytic_list.append(wordcounts)
        spkr_analytic_list.append(analytics.direct_args_from_others(xaif))
        spkr_analytic_list.append(analytics.indirect_args_from_others(xaif))

        spkr_analytic_list.append(analytics.loc_counts(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.arg_word_densities(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.arg_loc_densities(xaif, speaker=speaker))
        
        spkr_analytic_list.append(analytics.arg_intros(xaif))
        spkr_analytic_list.append(analytics.concl_first_perc(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.avg_arg_breadths(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.avg_arg_depths(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_serial(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_convergent(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_divergent(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_in_linked(xaif, speaker=speaker))

        spkr_analytic_list.append(analytics.premise_count(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.concl_count(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.prem_concl_ratio(xaif, speaker=speaker))
        spkr_analytic_list.append(analytics.ra_ca_ratio(xaif, speaker=speaker))

    
        if spkr_analytic_list != []:
            for s in spkr_rel_counts.keys():
                # rel_counts[s].update(concl_first[s])
                for a in spkr_analytic_list:
                    spkr_rel_counts[s].update(a[s])
                

    # Global analytics
    global_analytics = {}
    global_analytics = global_analytics | analytics.map_wordcount(xaif)
    global_analytics = global_analytics | analytics.loc_counts(xaif, speaker=False)
    global_analytics = global_analytics | analytics.arg_word_densities(xaif, speaker=False)
    global_analytics = global_analytics | analytics.arg_loc_densities(xaif, speaker=False)

    global_analytics = global_analytics | analytics.ra_in_serial(xaif, speaker=False)
    global_analytics = global_analytics | analytics.ra_in_convergent(xaif, speaker=False)
    global_analytics = global_analytics | analytics.ra_in_divergent(xaif, speaker=False)
    global_analytics = global_analytics | analytics.ra_in_linked(xaif, speaker=False)

    global_analytics = global_analytics | analytics.premise_count(xaif, speaker=False)
    global_analytics = global_analytics | analytics.concl_count(xaif, speaker=False)
    global_analytics = global_analytics | analytics.prem_concl_ratio(xaif, speaker=False)
    global_analytics = global_analytics | analytics.ra_ca_ratio(xaif, speaker=False)

    global_analytics = global_analytics | analytics.avg_inode_sentiment(xaif)
    global_analytics = global_analytics | analytics.arg_struct_sentiment(xaif)
    global_analytics = global_analytics | analytics.avgTenseScores(xaif)
    global_analytics = global_analytics | analytics.arg_struct_ner_types(xaif)
    


    #Adding analytics which calculate 'per node'
    if node_level:
        node_analytic_list = []
        node_analytic_list.append(analytics.node_wc(xaif))
        node_analytic_list.append(analytics.supportedNodes(xaif))
        node_analytic_list.append(analytics.attackedNodes(xaif))
        node_analytic_list.append(analytics.nodeTenseScores(xaif))
        node_analytic_list.append(analytics.ner(xaif))
        node_analytic_list.append(analytics.sentiment(xaif))
        print(node_analytic_list)

        # print(node_analytic_list)


        #Forecast-specific analytics
    if forecast:
        print(xaif)
        print("in forecast if")
        forecast_analytics_list = []
        forecast_analytics_list.append(analytics.addForecastAccuracy(xaif))
        forecast_analytics_list.append(analytics.addNodeOutcomes(xaif))
        subgraphs = analytics.getHypSubgraphs(xaif)
        print("subgraphs:")
        print(subgraphs)
        # forecast_analytics_list.append(analytics.raCount(xaif))
        i = 0
        for graph in subgraphs:
            i +=1
            subgraph_list = []
            print(graph)
            subgraph_list.append(analytics.raCount(graph))
            subgraph_list.append(analytics.caCount(graph))
            subgraph_list.append(analytics.forecast_wc(graph))
            subgraph_list.append(analytics.max_ra_chain(graph, speaker=False))
            subgraph_list.append(analytics.max_ca_chain(graph, speaker=False))
            # RA types
            subgraph_list.append(analytics.ra_in_linked(graph, speaker=False))
            subgraph_list.append(analytics.ra_in_convergent(graph, speaker=False))
            subgraph_list.append(analytics.ra_in_divergent(graph, speaker=False))
            subgraph_list.append(analytics.ra_in_serial(graph, speaker=False))
            # CA types
            subgraph_list.append(analytics.ca_undercut(graph, speaker=False))
            subgraph_list.append(analytics.ca_rebut(graph, speaker=False, verbose=False, skip_altgive=True))
            
            subgraph_list.append(analytics.restating_count(graph, speaker=False))
            
            subgraph_list.append(analytics.premise_count(graph, speaker=False))
            subgraph_list.append(analytics.concl_count(graph, speaker=False))
            subgraph_list.append(analytics.arg_word_densities(graph, speaker=False, verbose=False, skip_altgive=True))
            subgraph_list.append(analytics.ra_ca_ratio(graph, speaker=False))
            subgraph_list.append(analytics.prem_concl_ratio(graph, speaker=False))

            # subgraph_list.append(analytics.map_wordcount(graph))
            forecast_analytics_list.append({("Hypothesis " + str(i)) : subgraph_list})


        # STANDARD DEVIATION
        sd_df = pd.DataFrame
        column_names = []
        index = 0
        print(forecast_analytics_list)
        for values in forecast_analytics_list:
                if values:
                    item_pairs = values.items()
                    # print(item_pairs)

                    if len(values.keys()) != len(values.values()):
                        print("Lengths not equal")
                
                    for key, value in item_pairs:
                        print(key)
                        print(value)
                        row = []
                        if "Hypothesis" in key:
                            print("we got a hypothesis")

                            for item in value:
                                analytic_pair = item.items()
                                for key2, value2 in analytic_pair:
                                    # print(key2)
                                    # print(value2)

                                    # Only do once at start
                                    # print("index: " + str(index))
                                    if index == 0:
                                        column_names.append(key2)
                                        # print(key2)
                                    row.append(value2)
                                    # print(value2)
                            index += 1

                        if sd_df.empty == False:
                            new_row = dict(zip(column_names, row))
                            new_row_df = pd.DataFrame(new_row, index=[key])
                            sd_df = pd.concat([sd_df,new_row_df])
                            # df = pd.concat([df,new_row_df], ignore_index=True)
                        else:
                            data = dict(zip(column_names, row))
                            sd_df = pd.DataFrame(data, index=[key])
                            # df = pd.DataFrame(map_id:row, columns=column_names)
                    
                    print(sd_df)
                    sd_df.to_csv('out.csv', index=False)  


        ## TO ADD: FOR EACH COLUMN, RUN STANDARD DEVIATION SCRIPT
        for (columnName, columnData) in sd_df.items():
            print('Column Name : ', columnName)
            print('Column Contents : ', columnData.values)

            values_list = []
            for num in columnData.values:
                values_list.append(float(num))

            print(values_list)
            result = statistics.stdev(values_list)
            print({columnName + "_sd": result})
            forecast_analytics_list.append({columnName + "_sd": result})





    

            
                            
                    

                        




    #     analytics_all_hypotheses = []
    #     analytic_array = []
    #     index = 1
    #     for item in forecast_analytics_list:
    #         res = [val for key, val in item.items() if "Hypothesis" in key]
 
    #         # printing result
    #         print("Values for substring keys : " + str(res))
            
    #         analytic_array = []
    #         print(res)
    #         for list in res:
    #             for analytic in list:

    #                 for x in analytic.values():
    #                     analytic_array.append(x)
    #             analytics_all_hypotheses.append({index: analytic_array})
    #             index += 1
    # print(analytics_all_hypotheses)

    # print("no of lists = " + str(len(analytics_all_hypotheses)))

                    # for each in analytic:
                    #     print(each)

                




    
    xaif['analytics'] = {}
    xaif['analytics']['global'] = global_analytics

    if speaker:
        xaif['analytics'] = {"speaker": spkr_rel_counts}
    
    if node_level:
        xaif['analytics']['node'] = node_analytic_list
    if forecast:
        xaif['analytics']['forecast'] = forecast_analytics_list
    
    # print(xaif)
    return xaif
