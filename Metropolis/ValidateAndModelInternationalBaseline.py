from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE=Path(r'D:\PhD\Metropolis')
INFILE=BASE/'international_prematch_features.csv'
OUTM=BASE/'international_baseline_fold_metrics.csv'
OUTP=BASE/'international_baseline_predictions.csv'
OUTV=BASE/'international_feature_validation.csv'

df=pd.read_csv(INFILE,low_memory=False)
df['date']=pd.to_datetime(df['date'],errors='coerce'); df['year']=df.date.dt.year
classes=['H','D','A']; checks=[]
def add(n,p,d): checks.append({'check':n,'passed':bool(p),'detail':str(d)})
add('row_count',len(df)==49520,len(df))
add('target_valid',df.target_result.isin(classes).all(),df.target_result.value_counts().to_dict())
add('elo_complete',df[['home_elo_pre','away_elo_pre','elo_diff_pre']].notna().all().all(),df[['home_elo_pre','away_elo_pre']].isna().sum().to_dict())
for stem in ['points_last5','points_last10','gd_last5','gd_last10','comp_points_last5','comp_points_last10','comp_gd_last5','comp_gd_last10','matches_prev30','matches_prev90']:
    a,b,d='home_'+stem,'away_'+stem,stem+'_diff'
    if all(c in df for c in [a,b,d]):
        calc=df[a]-df[b]; mask=calc.notna()&df[d].notna(); err=(calc[mask]-df.loc[mask,d]).abs().max() if mask.any() else 0
        add('difference_'+stem,err<1e-9,f'max_abs_error={err}')
pd.DataFrame(checks).to_csv(OUTV,index=False)

elo_num=['elo_diff_pre']
rich_num=elo_num+[c for c in ['points_last5_diff','points_last10_diff','gf_last5_diff','gf_last10_diff','ga_last5_diff','ga_last10_diff','gd_last5_diff','gd_last10_diff','comp_points_last5_diff','comp_points_last10_diff','comp_gd_last5_diff','comp_gd_last10_diff','matches_prev30_diff','matches_prev90_diff','home_days_since_last_match','away_days_since_last_match','home_career_matches_before','away_career_matches_before','importance'] if c in df]
cat=[c for c in ['tournament_category','neutral','era'] if c in df]

def brier(y,p):
    Y=np.column_stack([(y==c).astype(float) for c in classes]); return np.mean(np.sum((p-Y)**2,axis=1))

def run(train,test,num,cats,label,fold):
    prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cats)],remainder='drop')
    model=Pipeline([('prep',prep),('clf',LogisticRegression(max_iter=3000,C=1.0))])
    model.fit(train[num+cats],train.target_result)
    p0=model.predict_proba(test[num+cats]); mc=list(model.named_steps['clf'].classes_); p=np.column_stack([p0[:,mc.index(c)] for c in classes]); pred=np.array(classes)[p.argmax(1)]
    m={'fold':fold,'model':label,'train_n':len(train),'test_n':len(test),'log_loss':log_loss(test.target_result,p,labels=classes),'brier_multiclass':brier(test.target_result.values,p),'accuracy':accuracy_score(test.target_result,pred),'mean_p_H':p[:,0].mean(),'mean_p_D':p[:,1].mean(),'mean_p_A':p[:,2].mean(),'actual_H':(test.target_result=='H').mean(),'actual_D':(test.target_result=='D').mean(),'actual_A':(test.target_result=='A').mean()}
    po=test[['date','home_team','away_team','target_result']].copy(); po['fold']=fold;po['model']=label;po['p_H']=p[:,0];po['p_D']=p[:,1];po['p_A']=p[:,2];po['prediction']=pred
    return m,po

metrics=[]; preds=[]
for fname,te,ty in [('train_to_2014_test_2018',2014,2018),('train_to_2018_test_2022',2018,2022)]:
    train=df[df.year<=te].copy(); test=df[df.year==ty].copy(); print(f'\n{fname}: train={len(train)}, test={len(test)}')
    for label,num,cats in [('elo_only',elo_num,[]),('rich_baseline',rich_num,cat)]:
        m,p=run(train,test,num,cats,label,fname); metrics.append(m);preds.append(p);print(f"  {label:14s} logloss={m['log_loss']:.4f} brier={m['brier_multiclass']:.4f} accuracy={m['accuracy']:.4f}")
pd.DataFrame(metrics).to_csv(OUTM,index=False); pd.concat(preds,ignore_index=True).to_csv(OUTP,index=False)
print('\n=== VALIDATION ===');print(pd.DataFrame(checks).to_string(index=False));print('\n=== FOLD METRICS ===');print(pd.DataFrame(metrics).to_string(index=False));print('\nCreated:');print(OUTM);print(OUTP);print(OUTV)