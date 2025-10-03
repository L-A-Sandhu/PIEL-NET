


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import tf2onnx
import onnx
import onnxruntime as ort

class TransposeLayer(layers.Layer):
    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm
        
    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)
    
    def get_config(self):
        config = super().get_config()
        config.update({'perm': self.perm})
        return config

class HybridModel(tf.keras.Model):
    def __init__(self, input_shape, pi_dim, checkpoint_path, 
                 loss_type='focal', alpha=0.2, beta=2.0, gamma=5.0, 
                 eta=0.1, focal_gamma=5.0):
        super(HybridModel, self).__init__()
        self.input_shape = input_shape
        self.pi_dim = pi_dim
        self.checkpoint_path = checkpoint_path
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.gamma_val = gamma
        self.eta = eta
        self.focal_gamma = focal_gamma
        self.model = self._build_model()
        self.onnx_model_path = None
        self.ort_session = None
        
        # Create custom metric that uses only the first pi_dim dimensions
        pi_dim = self.pi_dim
        def custom_mae(y_true, y_pred):
            y_true_labels = y_true[:, :pi_dim]
            return tf.reduce_mean(tf.abs(y_pred - y_true_labels), axis=-1)
        
        # Compile with selected loss and metric
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,  # Default is 0.001 - adjust this!
            clipvalue=0.5,        # Gradient clipping value
            beta_1=0.9,           # Momentum decay (default)
            beta_2=0.999,         # RMSprop decay (default)
            epsilon=1e-07         # Numerical stability
        )

        self.compile(optimizer='adam', 
                    loss=self._select_loss(), 
                    metrics=[custom_mae])

    def _select_loss(self):
        """Select loss function based on loss_type"""
        if self.loss_type == 'focal':
            return self.custom_mse_loss()
        elif self.loss_type == 'MSE':
            return self.mse_loss()
        elif self.loss_type == 'MAE':
            return self.mae_loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def custom_mse_loss(self):
        def loss(y_true, y_pred):
            y_true_labels = y_true[:, :self.pi_dim]
            Err_val = y_true[:, self.pi_dim]
            fuzzy_weights_val = y_true[:, self.pi_dim+1:]
            
            base_loss = tf.reduce_mean(tf.square(y_pred - y_true_labels), axis=-1)
            
            # Normalize weights to prevent explosion
            w_i = (self.alpha * fuzzy_weights_val[:, 0] + 
                   self.beta * fuzzy_weights_val[:, 1] + 
                   self.gamma_val * fuzzy_weights_val[:, 2])
            w_i = w_i / (tf.reduce_max(w_i) + 1e-8)  # Avoid division by zero
            
            # Clip large values to prevent exp(-large) → 0
            stability_term = tf.clip_by_value(base_loss + self.eta * Err_val, 
                                              -50.0, 50.0)  # Prevents underflow
            conf = tf.exp(-stability_term)
            
            # Clip conf to avoid NaN in pow()
            conf = tf.clip_by_value(conf, 1e-7, 1.0-1e-7)
            focal_term = tf.pow(1.0 - conf, self.focal_gamma)
            
            per_sample_loss = w_i * focal_term * base_loss
            return tf.reduce_mean(per_sample_loss)
        return loss

    def mse_loss(self):
        """Standard MSE loss using only true labels"""
        def loss(y_true, y_pred):
            y_true_labels = y_true[:, :self.pi_dim]
            return tf.reduce_mean(tf.square(y_pred - y_true_labels))
        return loss

    def mae_loss(self):
        """Standard MAE loss using only true labels"""
        def loss(y_true, y_pred):
            y_true_labels = y_true[:, :self.pi_dim]
            return tf.reduce_mean(tf.abs(y_pred - y_true_labels))
        return loss

    def _build_base_model(self):
        """Build the base model (feature extractor)"""
        inputs = layers.Input(shape=self.input_shape)
        conv3d = layers.Conv3D(32, (1, 1, 1), activation='elu', padding='same', 
                               kernel_initializer="glorot_uniform", bias_initializer="zeros")(inputs)
        conv_lstm1 = layers.ConvLSTM2D(16, (3, 3), activation='hard_sigmoid', 
                                       recurrent_activation='tanh', return_sequences=True, padding='same',
                                       kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", 
                                       bias_initializer="zeros")(conv3d)
        conv_lstm2 = layers.ConvLSTM2D(32, (3, 3), activation='hard_sigmoid', 
                                       recurrent_activation='tanh', return_sequences=True, padding='same',
                                       kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", 
                                       bias_initializer="zeros")(conv_lstm1)
        residual1 = layers.Add()([conv3d, conv_lstm2])
        conv_lstm3 = layers.ConvLSTM2D(16, (3, 3), activation='hard_sigmoid', 
                                       recurrent_activation='tanh', return_sequences=True, padding='same',
                                       kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", 
                                       bias_initializer="zeros")(residual1)
        conv_lstm4 = layers.ConvLSTM2D(32, (3, 3), activation='hard_sigmoid', 
                                       recurrent_activation='tanh', return_sequences=True, padding='same',
                                       kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", 
                                       bias_initializer="zeros")(conv_lstm3)
        residual2 = layers.Add()([conv_lstm4, residual1])
        permuted = TransposeLayer(perm=[0, 4, 2, 3, 1])(residual2)
        conv3d_final = layers.Conv3D(32, (1, 1, 1), activation='elu', padding='same',
                                     kernel_initializer="glorot_uniform", bias_initializer="zeros")(permuted)
        flat = layers.Flatten()(conv3d_final)
        dense = layers.Dense(self.pi_dim, name='base_output')(flat)
        return models.Model(inputs=inputs, outputs=dense, name='feature_extractor')

    def _build_model(self):
        """Build full model with one input"""
        main_input = layers.Input(shape=self.input_shape, name='main_input')
        base_output = self._build_base_model()(main_input)
        return models.Model(inputs=main_input, outputs=base_output)

    def call(self, inputs):
        return self.model(inputs)

    def fit(self, X_train, Y_train, validation_data=None, epochs=10, batch_size=32, patience=5):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        model_path = os.path.join(self.checkpoint_path, 'best_model.keras')
        self.onnx_model_path = os.path.join(self.checkpoint_path, 'model.onnx')
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        checkpoint = callbacks.ModelCheckpoint(
            model_path, 
            monitor='val_loss', 
            save_best_only=True,
            save_weights_only=False
        )
        
        history = super().fit(
            X_train, 
            Y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Convert to ONNX after training
        self.convert_to_onnx()
        return history

    def convert_to_onnx(self):
        """Convert TensorFlow model to ONNX format"""
        input_signature = [tf.TensorSpec(shape=(None, *self.input_shape), 
                          dtype=tf.float32, name='main_input')]
        
        # Convert using tf2onnx
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=[input_signature],
            output_path=self.onnx_model_path,
            opset=13
        )
        
        # Validate ONNX model
        try:
            onnx_model = onnx.load(self.onnx_model_path)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(f"ONNX model validation failed: {e}")
            raise

    def load_onnx_session(self):
        """Load ONNX runtime session"""
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError("ONNX model not found. Train model first.")
        
        self.ort_session = ort.InferenceSession(
            self.onnx_model_path,
            providers=['CPUExecutionProvider']
        )

    def predict(self, X_input, use_onnx=True):
        """Predict using TensorFlow or ONNX runtime"""
        if use_onnx:
            if self.ort_session is None:
                self.load_onnx_session()
                
            input_name = self.ort_session.get_inputs()[0].name
            onnx_input = {input_name: X_input.astype(np.float32)}
            return self.ort_session.run(None, onnx_input)[0]
        else:
            return self.model.predict(X_input)

    def summary(self):
        return self.model.summary()