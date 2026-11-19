
 // Trabalho 03 - Detector de Parafusos
// Modelo treinado no Google Colab.
// Executado no navegador com ONNX Runtime Web.

let SESSAO = null, INFO = null;

// ------------------------------------------------------------
// 1. Preparar a imagem
// ------------------------------------------------------------
function preparar(fonte, T, tarefa) {
    const w = fonte.videoWidth || fonte.naturalWidth || fonte.width;
    const h = fonte.videoHeight || fonte.naturalHeight || fonte.height;

    const cv = document.createElement('canvas');
    cv.width = T;
    cv.height = T;

    const g = cv.getContext('2d');

    let escala, dx, dy;

    if (tarefa === 'classificar') {
        // Recorta o quadrado central da imagem.
        const m = Math.min(w, h);

        g.drawImage(
            fonte,
            (w - m) / 2,
            (h - m) / 2,
            m,
            m,
            0,
            0,
            T,
            T
        );

        escala = T / m;
        dx = -(w - m) / 2 * escala;
        dy = -(h - m) / 2 * escala;

    } else {
        // LUGAR D: redimensionamento proporcional (letterbox).
        escala = Math.min(T / w, T / h);

        dx = (T - w * escala) / 2;
        dy = (T - h * escala) / 2;

        // Preenche o espaço restante com cinza.
        g.fillStyle = 'rgb(114,114,114)';
        g.fillRect(0, 0, T, T);

        g.drawImage(
            fonte,
            dx,
            dy,
            w * escala,
            h * escala
        );
    }

    const px = g.getImageData(0, 0, T, T).data;
    const tensor = new Float32Array(3 * T * T);

    // LUGAR C: normalização dos pixels entre 0 e 1.
    for (let i = 0; i < T * T; i++) {
        tensor[i] = px[4 * i] / 255;
        tensor[T * T + i] = px[4 * i + 1] / 255;
        tensor[2 * T * T + i] = px[4 * i + 2] / 255;
    }

    return { tensor, escala, dx, dy, w, h };
}

// ------------------------------------------------------------
// 2. Classificar
// ------------------------------------------------------------
function decidir(saida, classes, limiar) {
    let k = 0;

    for (let i = 1; i < saida.length; i++) {
        if (saida[i] > saida[k]) {
            k = i;
        }
    }

    // LUGAR G: tratamento de previsões com baixa confiança.
    return {
        classe: saida[k] >= limiar
            ? classes[k]
            : 'não tenho certeza',
        prob: saida[k]
    };
}

// ------------------------------------------------------------
// 3. Detectar
// ------------------------------------------------------------
function caixas(saida, nClasses, limiar) {
    const N = saida.length / (4 + nClasses);
    const lista = [];

    // LUGAR E: interpretação correta da saída do YOLO.
    for (let i = 0; i < N; i++) {
        let k = 0;
        let nota = saida[4 * N + i];

        for (let c = 1; c < nClasses; c++) {
            const prob = saida[(4 + c) * N + i];

            if (prob > nota) {
                nota = prob;
                k = c;
            }
        }

        // Ignora detecções abaixo do limiar de confiança.
        if (nota < limiar) continue;

        const cx = saida[i];
        const cy = saida[N + i];
        const bw = saida[2 * N + i];
        const bh = saida[3 * N + i];

        lista.push({
            x0: cx - bw / 2,
            y0: cy - bh / 2,
            x1: cx + bw / 2,
            y1: cy + bh / 2,
            nota,
            classe: k
        });
    }

    return lista;
}

// Calcula a sobreposição entre duas caixas.
function iou(a, b) {
    const ix = Math.max(
        0,
        Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)
    );

    const iy = Math.max(
        0,
        Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0)
    );

    const inter = ix * iy;

    const uniao =
        (a.x1 - a.x0) * (a.y1 - a.y0) +
        (b.x1 - b.x0) * (b.y1 - b.y0) -
        inter;

    return uniao > 0 ? inter / uniao : 0;
}

// LUGAR F: remoção de detecções duplicadas (NMS).
function nms(lista, limite) {
    const ordenadas = [...lista].sort(
        (a, b) => b.nota - a.nota
    );

    const resultado = [];

    for (const caixa of ordenadas) {
        const repetida = resultado.some(
            outra =>
                outra.classe === caixa.classe &&
                iou(outra, caixa) > limite
        );

        if (!repetida) {
            resultado.push(caixa);
        }
    }

    return resultado;
}

// LUGAR D: reposiciona as caixas na imagem original.
function desfazer(c, prep) {
    return {
        ...c,
        x0: (c.x0 - prep.dx) / prep.escala,
        y0: (c.y0 - prep.dy) / prep.escala,
        x1: (c.x1 - prep.dx) / prep.escala,
        y1: (c.y1 - prep.dy) / prep.escala
    };
}

// ------------------------------------------------------------
// 4. Carregar o modelo
// ------------------------------------------------------------
async function carregar(pasta = '.') {
    INFO = await (
        await fetch(
            pasta + '/modelo.json',
            { cache: 'no-store' }
        )
    ).json();

    SESSAO = await ort.InferenceSession.create(
        pasta + '/modelo.onnx',
        { executionProviders: ['wasm'] }
    );

    return INFO;
}

// ------------------------------------------------------------
// 5. Executar a inteligência artificial
// ------------------------------------------------------------
async function analisar(fonte) {
    const T = INFO.tamanho;

    const prep = preparar(
        fonte,
        T,
        INFO.tarefa
    );

    const entrada = new ort.Tensor(
        'float32',
        prep.tensor,
        [1, 3, T, T]
    );

    const t0 = performance.now();

    const r = await SESSAO.run({
        [SESSAO.inputNames[0]]: entrada
    });

    const ms = performance.now() - t0;

    const saida = r[SESSAO.outputNames[0]].data;

    if (INFO.tarefa === 'classificar') {
        return {
            ms,
            ...decidir(
                saida,
                INFO.classes,
                INFO.limiar
            )
        };
    }

    const achadas = nms(
        caixas(
            saida,
            INFO.classes.length,
            INFO.limiar
        ),
        0.45
    ).map(c => desfazer(c, prep));

    return {
        ms,
        caixas: achadas,
        w: prep.w,
        h: prep.h
    };
}

// Exportação das funções para testes.
if (typeof module !== 'undefined') {
    module.exports = {
        preparar,
        decidir,
        caixas,
        iou,
        nms,
        desfazer
    };
}
