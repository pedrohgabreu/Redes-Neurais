import torch

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    """
    Treina a rede neural e acompanha o Loss de treino e validação.
    Retorna o histórico para podermos plotar os gráficos exigidos no trabalho.
    """
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # --- MODO TREINAMENTO ---
        model.train() 
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()           # 1. Zera os gradientes antigos
            outputs = model(X_batch)        # 2. Faz a previsão (Forward)
            loss = criterion(outputs, y_batch) # 3. Calcula o erro
            loss.backward()                 # 4. Calcula os gradientes (Backward)
            optimizer.step()                # 5. Atualiza os pesos da rede
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- MODO AVALIAÇÃO (Validação) ---
        model.eval() 
        val_loss = 0.0
        
        with torch.no_grad(): # Desliga os gradientes para economizar memória e processamento
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(test_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Imprime o progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    return model, history