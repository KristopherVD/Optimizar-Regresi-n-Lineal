import numpy as np
import matplotlib.pyplot as plt

class RegresionOptimizador:
    def __init__(self, x_data, y_data):
        self.x_orig = np.array(x_data)
        self.y_orig = np.array(y_data)
        
        # 1. Centrado de datos
        self.x_mean = np.mean(self.x_orig)
        self.y_mean = np.mean(self.y_orig)
        self.x_cent = self.x_orig - self.x_mean
        self.y_cent = self.y_orig - self.y_mean

    def sse(self, beta_1):
        """Función objetivo: Sum of Squared Errors (SSE)"""
        predicciones = beta_1 * self.x_cent
        errores = self.y_cent - predicciones
        return np.sum(errores**2)

    def optimizar_seccion_aurea(self, a, b, tol=0.001):
        """Encuentra el beta_1 óptimo usando la Sección Áurea"""
        tau = 0.381966  # Constante áurea
        
        beta_1_izq = a + tau * (b - a)
        beta_1_der = b - tau * (b - a)
        
        f_izq = self.sse(beta_1_izq)
        f_der = self.sse(beta_1_der)
        
        while abs(b - a) > tol:
            if f_izq < f_der:
                b = beta_1_der
                beta_1_der = beta_1_izq
                f_der = f_izq
                beta_1_izq = a + tau * (b - a)
                f_izq = self.sse(beta_1_izq)
            else:
                a = beta_1_izq
                beta_1_izq = beta_1_der
                f_izq = f_der
                beta_1_der = b - tau * (b - a)
                f_der = self.sse(beta_1_der)
                
        # Retorna el beta_1 óptimo y su error correspondiente
        beta_optimo = (a + b) / 2
        return beta_optimo, self.sse(beta_optimo)

    def graficar_entregables(self):
        """Genera las gráficas solicitadas en la rúbrica"""
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- Gráfica 1: Comparativa de Pendientes ---
        axs[0].scatter(self.x_cent, self.y_cent, color='black', label='Datos Centrados', zorder=5)
        
        # Evaluando distintas pendientes (0.1, 0.3, 0.5, 0.7, 0.9)
        x_vals = np.array([min(self.x_cent)-5, max(self.x_cent)+5])
        pendientes_prueba = [0.1, 0.3, 0.5, 0.7, 0.9]
        colores = ['lightblue', 'cyan', 'dodgerblue', 'blue', 'darkblue']
        
        for beta, color in zip(pendientes_prueba, colores):
            axs[0].plot(x_vals, beta * x_vals, color=color, alpha=0.6, label=f'β1 = {beta}')
            
        beta_opt, _ = self.optimizar_seccion_aurea(0, 1)
        axs[0].plot(x_vals, beta_opt * x_vals, color='red', linewidth=2.5, label=f'Óptima (β1={beta_opt:.4f})')
        
        axs[0].set_title('Comparativa de Pendientes en Datos Centrados')
        axs[0].set_xlabel("Metros Cuadrados Centrados (x')")
        axs[0].set_ylabel("Renta Centrada (y')")
        axs[0].axhline(0, color='gray', linewidth=0.5)
        axs[0].axvline(0, color='gray', linewidth=0.5)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # --- Gráfica 2: Paisaje del Error (SSE) ---
        betas = np.arange(0, 1.1, 0.1) # Pasos de 0.1 según rúbrica
        errores = [self.sse(b) for b in betas]
        
        # Curva suave para mejor visualización
        betas_suave = np.linspace(0, 1, 100)
        errores_suave = [self.sse(b) for b in betas_suave]
        
        axs[1].plot(betas_suave, errores_suave, color='purple', label='Curva SSE')
        axs[1].scatter(betas, errores, color='black', zorder=5, label='Evaluaciones (Paso 0.1)')
        
        # Marcamos el mínimo encontrado
        sse_opt = self.sse(beta_opt)
        axs[1].scatter(beta_opt, sse_opt, color='red', s=100, zorder=6, label=f'Mínimo: {sse_opt:.2f}')
        
        axs[1].set_title('Paisaje del Error: SSE vs Pendiente (β1)')
        axs[1].set_xlabel('Valor de la Pendiente (β1)')
        axs[1].set_ylabel('Suma de Errores al Cuadrado (SSE)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ==========================================
# Ejecución del Reto
# ==========================================
if __name__ == "__main__":
    # Dataset proporcionado en Moodle
    x = [30, 35, 45, 50, 60, 70, 80, 90, 100, 120]
    y = [8.5, 10.2, 13.0, 14.5, 18.2, 20.1, 23.5, 26.2, 30.5, 35.0]
    
    modelo = RegresionOptimizador(x, y)
    
    # Resolviendo el problema de optimización [0, 1]
    beta_1_opt, sse_minimo = modelo.optimizar_seccion_aurea(0, 1, tol=0.001)
    
    # Calculando Beta 0 (para el reporte)
    beta_0_opt = modelo.y_mean - beta_1_opt * modelo.x_mean
    
    print("=== RESULTADOS DE OPTIMIZACIÓN ===")
    print(f"Beta 1 (Pendiente óptima): {beta_1_opt:.5f}")
    print(f"Beta 0 (Intersección):     {beta_0_opt:.5f}")
    print(f"SSE Mínimo alcanzado:      {sse_minimo:.5f}")
    print("----------------------------------")
    print("Generando gráficas obligatorias...")
    
    modelo.graficar_entregables()