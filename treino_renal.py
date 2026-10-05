import json, re, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
path=sys.argv[1]
cols=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
rows=[]; data=False
for line in open(path,encoding='latin-1'):
    s=line.strip()
    if not s or s.startswith('%'): continue
    if s.lower().startswith('@data'): data=True; continue
    if not data: continue
    vals=[v.strip() for v in s.split(',')]
    vals=[v for v in vals if v!='']  # vírgulas duplicadas
    if len(vals)!=25: print('linha ignorada',len(vals),s[:80]); continue
    rows.append(vals)
df=pd.DataFrame(rows,columns=cols).replace('?',np.nan)
binmap={'normal':0,'abnormal':1,'notpresent':0,'present':1,'no':0,'yes':1,'good':0,'poor':1}
cat=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
for c in cat: df[c]=df[c].map(lambda v: binmap.get(v,np.nan) if isinstance(v,str) else v)
num=[c for c in cols if c not in cat+['class']]
for c in num: df[c]=pd.to_numeric(df[c],errors='coerce')
y=df['class'].map({'ckd':1,'notckd':0})
print('linhas',len(df),'classes',y.value_counts().to_dict(),'nulos de classe',y.isna().sum())
feats=cols[:-1]; X=df[feats].astype(float)
print('faltantes por variável',X.isna().sum().to_dict())
print('linhas completas',X.dropna().shape[0])
triagem=['age','bp','bgr','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
out={}
for nome,vs in [('completo',feats),('triagem',triagem)]:
    imp_kinds={v:('most_frequent' if v in cat else 'median') for v in vs}
    # imputação: mediana (numéricas) e moda (sim/não), calculadas só no treino
    fill={v:(Xtr[v].mode()[0] if v in cat else Xtr[v].median()) for v in vs}
    A=Xtr[vs].fillna(fill); B=Xte[vs].fillna(fill)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0))
    cv=cross_val_score(pipe,A,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='roc_auc')
    cva=cross_val_score(pipe,A,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='accuracy')
    pipe.fit(A,ytr); p=pipe.predict_proba(B)[:,1]; yp=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(yte,yp).ravel()
    sc,lr=pipe[0],pipe[1]
    m=dict(acc=accuracy_score(yte,yp),sens=recall_score(yte,yp),spec=tn/(tn+fp),ppv=precision_score(yte,yp) if tp+fp else float('nan'),auc=roc_auc_score(yte,p),cv_auc_mean=cv.mean(),cv_auc_std=cv.std(),cv_acc_mean=cva.mean(),tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))
    out[nome]=dict(vars=vs,fill={k:float(v) for k,v in fill.items()},mean=dict(zip(vs,sc.mean_.tolist())),scale=dict(zip(vs,sc.scale_.tolist())),coef=dict(zip(vs,lr.coef_[0].tolist())),intercept=float(lr.intercept_[0]),metrics=m,test=[[round(float(a),4),int(b)] for a,b in zip(p,yte)])
    print('\n==',nome,json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items()}))
    print(sorted({k:round(v,2) for k,v in zip(vs,lr.coef_[0])}.items(),key=lambda t:-abs(t[1])))
meta=dict(n=len(df),n_ckd=int(y.sum()),n_train=len(ytr),n_test=len(yte),n_complete=int(X.dropna().shape[0]),missing={k:int(v) for k,v in X.isna().sum().items()},ranges={c:[float(X[c].min()),float(X[c].max())] for c in num})
# exemplos do teste com valores originais (faltantes preenchidos pelo treino) 
fillall={v:(Xtr[v].mode()[0] if v in cat else Xtr[v].median()) for v in feats}
Xte_f=Xte.fillna(fillall)
json.dump(dict(models=out,meta=meta,Xte=Xte_f.round(3).values.tolist(),Xte_missing=Xte.isna().values.tolist(),yte=yte.tolist(),feats=feats),open('modelo_renal.json','w'))
