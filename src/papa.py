import json
from . import xaif_toolbox as ova3
from . import analytics
import sys
import os

def all_analytics(xaif):
    # This involves redundancy currently bc all analytics are designed to call the info collection
    # so that they can be run individually with just the XAIF...


    # Speaker-level analytics
    analytic_list = []
    rel_counts = analytics.arg_relation_counts(xaif)

    xaif = ova3.ova2_to_ova3(xaif)
    if 'AIF' in xaif.keys():
        wordcounts = ova3.spkr_wordcounts(xaif)
    analytic_list.append(wordcounts)
    analytic_list.append(analytics.concl_first_perc(xaif))
    analytic_list.append(analytics.arg_word_densities(xaif))
    analytic_list.append(analytics.arg_loc_densities(xaif))
    analytic_list.append(analytics.ra_in_serial(xaif))
    analytic_list.append(analytics.ra_in_convergent(xaif))
    analytic_list.append(analytics.ra_in_divergent(xaif))
    analytic_list.append(analytics.ra_in_linked(xaif))
    analytic_list.append(analytics.avg_arg_depths(xaif))
    analytic_list.append(analytics.avg_arg_breadths(xaif))
    analytic_list.append(analytics.arg_intros(xaif))
    analytic_list.append(analytics.direct_args_from_others(xaif))
    analytic_list.append(analytics.indirect_args_from_others(xaif))

    # concl_first = analytics.concl_first_perc(xaif)
    # arg_densities = analytics.arg_densities(xaif)

    for s in rel_counts.keys():
        # rel_counts[s].update(concl_first[s])
        for a in analytic_list:
            rel_counts[s].update(a[s])

    xaif['analytics'] = rel_counts
    return xaif
