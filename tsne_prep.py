import visual

<<<<<<< Updated upstream
w = visual.WicTsne()
w.set_model('output/vua-snli/checkpoints/snli_epoch20.pt', device='cuda')
w.embed(layer=2)
w.save_embeddings('output/vua-snli/evaluation/tsne_embeddings_snli_epoch20.pt')
