import visual

w = visual.WicTsne('output/vua-snli/checkpoints/snli_epoch20.pt',layer=2,device='cuda')
w.embed()
w.save_embeddings('ouput/vua-snli/evaluation/tsne_embeddings_snli_epoch20.pt')