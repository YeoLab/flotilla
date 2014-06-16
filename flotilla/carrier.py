__author__ = 'lovci'

"""

handles database connection, performs operations as a slave if run as a script.

"""



#!grep def slave.py
import random

from .frigate import *
from .cargo import *

mongoHost = ''
mongoPort = ''
min_cells = ''

def fill_database(mongodb, predictor, target, verbose=False):
    """
    mongodb - mongo database connection object
    predictor - X values, i.e. RPKMs.  Rows == cell ids, cols == gene ids
    target - Y values, i.e. PSIs.  Rows == cell ids, cols == exon ids
    """

    nRand = 10
    if verbose:
        sys.stderr.write("processing...\n")
    event_i = 0
    for splice_event in get_unstarted_events(mongodb):
        event_i += 1
        E = splice_event['eventId']
        if verbose:
            sys.stderr.write(E)

        X = predictor.copy()
        y = target.copy()

        y = y.ix[E].dropna()
        X, y = X.align(y, join='inner', axis=0)
        basic={'eventId':E, '_id':splice_event['_id']}

        #objects = basic.copy()
        series = basic.copy()
        lists = basic.copy()
        values = basic.copy()

        for let in letters + ['all']:
            if verbose:
                sys.stderr.write("%s started\n" % let)

            if len(let) == 1:
                XX, yy = [i.ix[[j for j in i.index if j.startswith(let)]] for i in [X,y]]

            elif len(let)==2:
                XX, yy = [i.ix[[j for j in i.index if j.startswith(let[0]) or \
                                j.startswith(let[1])]] for i in [X,y]]
            else:
                assert let == "all"
                XX, yy = X, y

            values['nCells_' + let] = len(yy)
            if verbose:
                print let, E, len(yy)
            if len(yy) <= min_cells:
                continue

            series['yy_' + let] = yy.to_json()

            regressor, oob_score_dist = get_regressor(XX,yy, verbose=verbose)
            series['importances_' + let] = regressor.feature_importances.to_json()
            values['clf_randomstate_' + let] = regressor.random_state

            boosting_regressor = get_boosting_regressor(XX,yy, verbose=verbose)
            values['boosting_clf_randomstate_' + let] = boosting_regressor.random_state
            series['boosting_importances_' + let] = regressor.feature_importances.to_json()

            dcor_DC, dcor_DR, dcor_DVX, dcor_DVY = apply_dcor(XX, yy)
            series['dcor_DC_' + let] = dcor_DC.to_json()
            series['dcor_DR_' + let] = dcor_DR.to_json()
            series['dcor_DVX_' + let] = dcor_DVX.to_json()
            series['dcor_DVY_' + let] = dcor_DVY.to_json()

            lists['scoreDist_' + let] = oob_score_dist

            values['score_' + let] = float(regressor.oob_score_)

            if np.mean(np.array(oob_score_dist)) > .05: #arbitrary... this is actually a pretty bad score.
                #don't waste time on random calculation if the real calcuations are crappy
                for i in range(nRand):
                    if verbose:
                        sys.stderr.write('shuffle\n')
                    rand_yy = pd.Series(random.sample(yy, len(yy)), index=yy.index)
                    regressor, oob_score_dist = get_regressor(XX, rand_yy, n_tries=3, verbose=verbose)
                    lists['rand_%d_scoreDist_' % i + let] = oob_score_dist

            for method, method_name in [(stats.pearsonr, 'pearson'),
                                        (stats.spearmanr, 'spearman'), ]:
                if verbose:
                    sys.stderr.write("trying method %s\n" % method_name)
                try:
                    r, p = apply_calc_rs(XX, yy, method=method)
                except TimeoutError:
                    sys.stderr.write("r caculation with method %s timed out on event %s\n" %(method_name, E))
                    continue

                series[method_name + "_corr_r_" + let] = r.to_json()
                series[method_name + "_corr_p_" + let] = p.to_json()

            if verbose:
                sys.stderr.write("%s finished\n" % let)

            try:
                robust_intercept, robust_slope, robust_t, robust_p = apply_calc_robust(XX, yy)
                series["robust_intercept_" + let] = robust_intercept.to_json()
                series["robust_slope_" + let] = robust_slope.to_json()
                series["robust_t_" + let] = robust_t.to_json()
                series["robust_p_" + let] = robust_p.to_json()

            except TimeoutError:
                sys.stderr.write("robust regression timed out on event %s\n" % E)
                continue
            try:
                slope = apply_calc_slope(XX, yy)
                series["slope_" + let] = slope.to_json()

            except TimeoutError:
                sys.stderr.write("lingregress timed out on event %s\n" % E)
                continue
        if verbose:
                sys.stderr.write("saving %s\n" % E)
        splice_event['finished'] = True
        mongodb['list'].save(splice_event)
        mongodb['values'].save(values)
        mongodb['series'].save(series)
        mongodb['lists'].save(lists)

    sys.stderr.write("database is full, checked %d events\n" % (event_i+1))


def get_mongo_db(db, mongoHost=mongoHost, mongoPort=mongoPort):
    from pymongo import MongoClient
    #ssh = subprocess.Popen(["ssh", "-L", ("%s:localhost:%s" %(mongoPort, mongoPort)), mongoHost, "-N"])
    c = MongoClient(mongoHost, port=mongoPort)
    sys.stderr.write('connected to database on %s:%d\n' % (mongoHost, mongoPort))
    return c, c[db]

def load_event_list(mongo, target, target_type="SE"):

    mongo['list'].drop()
    _ = [mongo['list'].save({"eventId":ev, "started":False, "finished":False,
                                 'splice_type':target_type}) for ev in target.index]

def reset_event_list(mongodb):
    mongodb['list'].update({}, {'$set':{'started':False, 'finished':False}}, multi=True)

def reset_event_data(mongodb):
    mongodb.drop_collection('values')
    mongodb.drop_collection('lists')
    mongodb.drop_collection('series')

def poll(mongodb):
    #must be refactored for new "letters", i.e. celltypes

    list_size = mongodb['list'].count()
    started_size = mongodb['list'].find({'started':True}).count()
    finished_size = mongodb['list'].find({'finished':True}).count()

    total_count = mongodb['values'].count()
#     all_regressor_count = mongodb['values'].find({'score_all':{'$exists':True}}).count()
#     N_regressor_count = mongodb['values'].find({'score_N':{'$exists':True}}).count()
#     M_regressor_count = mongodb['values'].find({'score_M':{'$exists':True}}).count()
#     P_regressor_count = mongodb['values'].find({'score_P':{'$exists':True}}).count()
#     S_regressor_count = mongodb['values'].find({'score_S':{'$exists':True}}).count()
#     PN_regressor_count = mongodb['values'].find({'score_PN':{'$exists':True}}).count()
#     NM_regressor_count = mongodb['values'].find({'score_NM':{'$exists':True}}).count()
#     MS_regressor_count = mongodb['values'].find({'score_MS':{'$exists':True}}).count()

    print "%d events in whole transcriptome"  %(list_size)
    print "%d (%.2f %%) events started"  %(started_size, 100*started_size/list_size)
    print "%d (%.2f %%) events finished"  %(finished_size, 100*finished_size/list_size)
    print "%d events examined"  %(total_count)
#     print "%d (%.2f%%) with P regressors" %(P_regressor_count, 100*P_regressor_count/total_count)
#     print "%d (%.2f%%) with N regressors" %(N_regressor_count, 100*N_regressor_count/total_count)
#     print "%d (%.2f%%) with M regressors" %(M_regressor_count, 100*M_regressor_count/total_count)
#     print "%d (%.2f%%) with S regressors" %(S_regressor_count, 100*S_regressor_count/total_count)
#     print "%d (%.2f%%) with PN regressors" %(PN_regressor_count, 100*PN_regressor_count/total_count)
#     print "%d (%.2f%%) with NM regressors" %(NM_regressor_count, 100*NM_regressor_count/total_count)
#     print "%d (%.2f%%) with MS regressors" %(MS_regressor_count, 100*MS_regressor_count/total_count)
#     print "%d (%.2f%%) with combined regressors" %(all_regressor_count, 100*all_regressor_count/total_count)


def begin(db = 'events'):
    (SEpsis, rpkm) = load_transcriptome_data()

    rbpRpkms = rpkms.ix[pd.Series(rbps.index).dropna()].fillna(0)

    mongo_con, mongodb = get_mongo_db(db)
    sys.stderr.write("finished loading raw study_data\n")
    return (SEpsis, rbpRpkms, mongodb)


# load_event_list(mongodb, SEpsis, target_type="SE")

# if __name__ == "__main__":
#     reset_event_list(mongodb)
#     reset_event_data(mongodb)


if __name__ == "__main__":
    (SEpsis, rpkm, mongodb) = begin(db='misoevents')
    fill_database(mongodb, rpkm, SEpsis, verbose=True)
    poll(mongodb)

#mongodb.values.find_one({'nCells_all':{'$gt':10}})

# mongodb.list.update({'started':True, 'finished':False}, {"$set":{"started":False}}, multi=True)