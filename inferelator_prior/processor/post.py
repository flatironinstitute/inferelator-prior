import sys
import pandas as pd
from tqdm import tqdm
from iteration_utilities import duplicates

class PostProcessor(Object):
  """ Handles post processing op on the prior matrix 
      including : - gene names conversion 
                  - removal of all zero rows
  """

  input_file = ''
  gtf_file = ''
  prior_df = None
  tled = None
  recoded = None
  nnz = 0

  def ens_tl(self,field):
    self.recoded = []
    ens = self.prior_df[field].values
    for ensid in tqdm(ens):
        x = ensid
        if x in self.tled:
           x = self[ensid]    
        recoded.append(x)
    self.prior_df[field].values = recoded
    return self

  def out(self,rec = '_rec.tsv'):
    self.prior_df.to_csv(input_file.replace('.tsv',rec),sep='\t',index=None)

  def read_prior(self):
    self.prior_df = pd.read_csv(input_file,sep='\t',header=0)
    self.tled = {}
    return self

  def process_gtf(self):
    gtf = [x.strip() for x in open(self.gtf_file).readlines()]
    ttf = list(set([g.split('gene_id')[1] for g in tqdm(gtf) if (('gene_id' in g) and ('gene_name' in g))]))

    for x in tqdm(ttf):
        x = x.split()
        key = x[0].replace('"','').replace(';','')
        val = x[x.index('gene_name')+1].replace('"','').replace(';','')
        if key not in self.tled.keys():
            self.tled[key] = val
    return self        

  def run(self,fields):
    self.read_prior().process_gtf()
    
    for f in fields:
      self.ens_tl(field=f)
    
    if self.nnz > 0:
      tmp = self.prior_df
      tmp.index = tmp.gene_name
      tmp = tmp.drop(columns='gene_name')
      tmp = tmp.groupby(tmp.index).agg(sum)
      tmp = tmp.loc[(tmp!=0).any(1)]
    self.out()
    
# prior['target'] = ens_tl(prior, 'target', tled)        
# prior['regulator'] = ens_tl(prior, 'regulator', tled)        


