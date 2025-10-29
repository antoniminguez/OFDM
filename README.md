# OFDM

import numpy as np

# --- 1. Parámetros del Sistema OFDM ---
N_FFT = 64         # Tamaño de la FFT (número total de subportadoras)
N_CP = 16          # Longitud del Prefijo Cíclico (Cyclic Prefix)
N_carriers = 48    # Número de subportadoras que usarán para datos
# (Dejamos algunas en los bordes y el centro (DC) vacías, como es habitual)

print(f"Iniciando simulación OFDM simple...")
print(f"Tamaño FFT: {N_FFT}, Prefijo Cíclico: {N_CP}, Portadoras de datos: {N_carriers}\n")

# --- 2. Transmisor (TX) ---

# a. Generar datos aleatorios (simulamos símbolos ya modulados, ej: QPSK)
# En un sistema real, aquí iría la codificación y modulación (BPSK, QPSK, 16-QAM)
tx_data_symbols = np.random.randn(N_carriers) + 1j * np.random.randn(N_carriers)

# b. Mapeo a las subportadoras de la IFFT
# Creamos el vector completo para la IFFT, poniendo los datos en su sitio
ifft_input_buffer = np.zeros(N_FFT, dtype=complex)

# Mapeo simple (centrado, saltando la portadora DC 0)
ifft_input_buffer[1 : N_carriers//2 + 1] = tx_data_symbols[:N_carriers//2]
ifft_input_buffer[N_FFT - N_carriers//2 :] = tx_data_symbols[N_carriers//2:]

print(f"[TX] Símbolos de datos generados: {len(tx_data_symbols)}")

# c. El corazón de OFDM: Transformada Inversa de Fourier (IFFT)
# Convertimos las subportadoras (frecuencia) a una señal en el dominio del tiempo
time_domain_signal = np.fft.ifft(ifft_input_buffer)

print("[TX] IFFT completada. Señal creada en dominio del tiempo.")

# d. Añadir Prefijo Cíclico (CP)
# Se copia el final de la señal al principio para combatir el multitrayecto
cp = time_domain_signal[-N_CP:]
ofdm_symbol_with_cp = np.concatenate([cp, time_domain_signal])

print(f"[TX] Prefijo Cíclico añadido. Longitud total del símbolo: {len(ofdm_symbol_with_cp)}")

# --- 3. Simulación del Canal ---
# En un sistema real, aquí la señal se enviaría por el aire,
# sufriendo ruido, atenuación y multitrayecto.
# Por simplicidad, asumimos un canal perfecto (solo TX -> RX).
received_signal = ofdm_symbol_with_cp

# --- 4. Receptor (RX) ---

print("\n--- Recepción ---")

# a. Quitar Prefijo Cíclico
# Descartamos la parte del CP que solo sirvió para la protección
signal_without_cp = received_signal[N_CP:]

print(f"[RX] Prefijo Cíclico eliminado. Longitud: {len(signal_without_cp)}")

# b. El corazón del receptor: Transformada de Fourier (FFT)
# Convertimos la señal de tiempo de vuelta al dominio de la frecuencia
fft_output_buffer = np.fft.fft(signal_without_cp)

print("[RX] FFT completada. Símbolos recuperados en frecuencia.")

# c. Demapeo de subportadoras
# Extraemos solo los símbolos de datos de las portadoras que usamos
rx_data_symbols = np.concatenate([
    fft_output_buffer[1 : N_carriers//2 + 1],
    fft_output_buffer[N_FFT - N_carriers//2 :]
])

print(f"[RX] Símbolos de datos extraídos: {len(rx_data_symbols)}")

# --- 5. Verificación ---
# Comprobamos si los datos recibidos son (casi) iguales a los enviados
# Usamos 'allclose' por los pequeños errores numéricos de la FFT/IFFT
if np.allclose(tx_data_symbols, rx_data_symbols):
    print("\nÉXITO: Los símbolos recibidos coinciden con los enviados (canal perfecto).")
else:
    print("\nFALLO: Los símbolos recibidos no coinciden.")
