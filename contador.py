import cv2
import numpy as np

MIN_AREA = 2500

COLOR_RANGES = {
    "vermelho": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (179, 255, 255))
    ],
    "verde": [((35, 80, 50), (85, 255, 255))],
    "azul":  [((90, 80, 50), (130, 255, 255))],
    "amarelo": [((20, 100, 80), (35, 255, 255))],
}

def count_color_objects(hsv, ranges):
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for (lower, upper) in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lower_np, upper_np))

    kernel = np.ones((5, 5), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= MIN_AREA:
            count += 1
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

    return count, boxes

def draw_panel(img, title, data, x, y, line_h=28):
    cv2.putText(img, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    yy = y + line_h
    for k, v in data.items():
        cv2.putText(img, f"{k}: {v}", (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        yy += line_h

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não consegui abrir a webcam.")
        return

    # Placares acumulados
    scoreboard = {k: 0 for k in COLOR_RANGES.keys()}

    # Guarda a contagem "na tela" anterior para detectar aumento (entrada)
    prev_live = {k: 0 for k in COLOR_RANGES.keys()}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        live_counts = {}
        overlay = frame.copy()

        # conta e desenha caixas
        for name, ranges in COLOR_RANGES.items():
            cnt, boxes = count_color_objects(hsv, ranges)
            live_counts[name] = cnt

            for (x, y, w, h) in boxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(overlay, name, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Atualiza placar: soma APENAS quando aumentou a quantidade na tela
        for k in live_counts.keys():
            diff = live_counts[k] - prev_live[k]
            if diff > 0:
                scoreboard[k] += diff
        prev_live = live_counts.copy()

        # Painéis: ao vivo e acumulado
        draw_panel(overlay, "AO VIVO (na tela)", live_counts, 10, 30)
        draw_panel(overlay, "PLACAR (acumulado)", scoreboard, 10, 190)

        cv2.putText(overlay, "Teclas: R=zerar placar | Q=sair",
                    (10, overlay.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Placar de placas por cor", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            scoreboard = {k: 0 for k in COLOR_RANGES.keys()}
            prev_live = {k: 0 for k in COLOR_RANGES.keys()}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
