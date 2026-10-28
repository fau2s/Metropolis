from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd


BASE = Path(r'D:\PhD\Metropolis'); INFILE=BASE/'results.csv'
OUT = BASE/'international_prematch_features.csv'; SUMMARY=BASE/'international_prematch_feature_summary.csv'
df = pd.read_csv(INFILE,low_memory=False)
req = ['date','home_team','away_team','home_score','away_score','tournament','neutral']
miss = [c for c in req if c not in df.columns]
if miss:
    raise RuntimeError(f'Missing columns: {miss}')
df['date'] = pd.to_datetime(df.date,errors='coerce'); df['home_score']=pd.to_numeric(df.home_score,errors='coerce'); df['away_score']=pd.to_numeric(df.away_score,errors='coerce')
df = df.dropna(subset=['date','home_team','away_team','home_score','away_score']).copy(); df['_row_id']=np.arange(len(df)); df=df.sort_values(['date','_row_id']).reset_index(drop=True)


def tinfo(x):
 s=str(x).lower()
 if 'friendly' in s:return 'friendly',0
 if any(k in s for k in ['qualification','qualifier','qualifying']):return 'qualification',1
 if 'fifa world cup' in s or s.strip()=='world cup':return 'world_cup',2
 if any(k in s for k in ['uefa euro','european championship','copa américa','copa america','africa cup of nations','african cup of nations','asian cup','gold cup','ofc nations cup']):return 'continental_final',2
 if 'nations league' in s:return 'nations_league',1
 return 'other_competitive',1
z=df.tournament.map(tinfo); df['tournament_category']=[x[0] for x in z]; df['importance']=[x[1] for x in z]; df['competitive']=(df.importance>0).astype(int); df['era']=(df.date.dt.year//10*10).astype(int)
ratings=defaultdict(lambda:1500.); hist=defaultdict(lambda:deque(maxlen=200)); chist=defaultdict(lambda:deque(maxlen=200))
def mn(x):return float(np.mean(x)) if x else np.nan
def hf(team,date,h):
 p=list(h[team]); o={}
 for n in (5,10):
  q=p[-n:]; o[f'points_last{n}']=mn([r['pts'] for r in q]); o[f'gf_last{n}']=mn([r['gf'] for r in q]); o[f'ga_last{n}']=mn([r['ga'] for r in q]); o[f'gd_last{n}']=mn([r['gf']-r['ga'] for r in q]); o[f'matches_available_last{n}']=len(q)
 if p:
  o['days_since_last_match']=(date-p[-1]['date']).days; o['matches_prev30']=sum((date-r['date']).days<=30 for r in p); o['matches_prev90']=sum((date-r['date']).days<=90 for r in p); o['career_matches_before']=len(p)
 else:o.update(days_since_last_match=np.nan,matches_prev30=0,matches_prev90=0,career_matches_before=0)
 return o
def cf(team):
 p=list(chist[team]);o={}
 for n in (5,10):
  q=p[-n:];o[f'comp_points_last{n}']=mn([r['pts'] for r in q]);o[f'comp_gd_last{n}']=mn([r['gf']-r['ga'] for r in q]);o[f'comp_matches_available_last{n}']=len(q)
 return o
records=[]
for date,day in df.groupby('date',sort=True):
 updates=[]
 for _,r in day.iterrows():
  h=str(r.home_team);a=str(r.away_team);neutral=bool(r.neutral) if not pd.isna(r.neutral) else False; he=ratings[h];ae=ratings[a];adv=0. if neutral else 60.; ex=1/(1+10**(-((he+adv)-ae)/400)); H=hf(h,date,hist);A=hf(a,date,hist);HC=cf(h);AC=cf(a);hs=float(r.home_score);ass=float(r.away_score)
  target='H' if hs>ass else 'A' if hs<ass else 'D'; actual=1. if hs>ass else 0. if hs<ass else .5
  rec=r.to_dict();rec.update(home_elo_pre=he,away_elo_pre=ae,elo_diff_pre=he-ae,elo_diff_with_home_adv_pre=he+adv-ae,home_elo_expected_win_component=ex,target_result=target)
  for k,v in H.items():rec['home_'+k]=v
  for k,v in A.items():rec['away_'+k]=v
  for k,v in HC.items():rec['home_'+k]=v
  for k,v in AC.items():rec['away_'+k]=v
  for stem in ['points_last5','points_last10','gf_last5','gf_last10','ga_last5','ga_last10','gd_last5','gd_last10','comp_points_last5','comp_points_last10','comp_gd_last5','comp_gd_last10','matches_prev30','matches_prev90']:rec[stem+'_diff']=rec.get('home_'+stem,np.nan)-rec.get('away_'+stem,np.nan)
  records.append(rec);K=20 if r.importance==0 else 40 if r.importance>=2 else 30;updates.append((h,a,K*(actual-ex),hs,ass,int(r.competitive)))
 for h,a,d,hs,ass,comp in updates:
  ratings[h]+=d;ratings[a]-=d;hp=3 if hs>ass else 1 if hs==ass else 0;ap=3 if ass>hs else 1 if hs==ass else 0;hr={'date':date,'gf':hs,'ga':ass,'pts':hp};ar={'date':date,'gf':ass,'ga':hs,'pts':ap};hist[h].append(hr);hist[a].append(ar)
  if comp:chist[h].append(hr);chist[a].append(ar)
out=pd.DataFrame(records).sort_values('_row_id').drop(columns=['_row_id']);out.to_csv(OUT,index=False)
summary=pd.DataFrame([{'matches':len(out),'date_min':out.date.min(),'date_max':out.date.max(),'teams':len(set(out.home_team)|set(out.away_team)),'target_H':int((out.target_result=='H').sum()),'target_D':int((out.target_result=='D').sum()),'target_A':int((out.target_result=='A').sum()),'same_day_policy':'features frozen before all matches on date; updates after date batch','elo_policy':'custom 1500 base; +60 non-neutral home advantage; K 20/30/40'}]);summary.to_csv(SUMMARY,index=False)
print('\n=== INTERNATIONAL PRE-MATCH FEATURE BUILD ===');print(summary.to_string(index=False));print('\nCreated:');print(OUT);print(SUMMARY)
