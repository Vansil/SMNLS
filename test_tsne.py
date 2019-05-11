import models
import visual

model = models.BaselineElmo()
word = 'carry'
path = 'results/wic_tsne_baselineelmo012_carry.png'

visual.wicTsne(model, word, path)