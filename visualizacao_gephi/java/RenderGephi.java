import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.prefs.AbstractPreferences;
import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

import org.gephi.graph.api.Column;
import org.gephi.graph.api.Edge;
import org.gephi.graph.api.Graph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.io.exporter.api.ExportController;
import org.gephi.io.exporter.preview.PDFExporter;
import org.gephi.io.exporter.preview.PNGExporter;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2;
import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2Builder;
import org.gephi.layout.plugin.noverlap.NoverlapLayout;
import org.gephi.layout.plugin.noverlap.NoverlapLayoutBuilder;
import org.gephi.preview.api.PreviewController;
import org.gephi.preview.api.PreviewModel;
import org.gephi.preview.api.PreviewProperty;
import org.gephi.preview.types.EdgeColor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.openide.util.Lookup;

/**
 * Renderizador headless com as bibliotecas do Gephi 0.10.1 instalado.
 * Uso: RenderGephi entrada.gexf prefixo_saida chave=valor...
 *   layout=fixo|fa2  iter=1500  linlog=true  scaling=2.0  gravity=1.0  threads=4  noverlap=true
 *   edgeOpacity=8  edgeThickness=0.3  labels=true  labelSize=10  png=4000
 * Saídas: prefixo.gephi, prefixo_preview.{pdf,svg,png}, prefixo_layout.gexf (coordenadas do Gephi).
 */
public class RenderGephi {

    /** Preferências só em memória: evita escrever no perfil do Gephi do usuário. */
    public static class EphemeralPreferences implements PreferencesFactory {
        private static final Preferences ROOT = new MemoryPreferences(null, "");
        public Preferences systemRoot() { return ROOT; }
        public Preferences userRoot() { return ROOT; }
    }

    static class MemoryPreferences extends AbstractPreferences {
        private final Map<String, String> values = new HashMap<>();
        private final Map<String, MemoryPreferences> children = new HashMap<>();
        MemoryPreferences(AbstractPreferences parent, String name) { super(parent, name); }
        protected void putSpi(String k, String v) { values.put(k, v); }
        protected String getSpi(String k) { return values.get(k); }
        protected void removeSpi(String k) { values.remove(k); }
        protected void removeNodeSpi() { }
        protected String[] keysSpi() { return values.keySet().toArray(new String[0]); }
        protected String[] childrenNamesSpi() { return children.keySet().toArray(new String[0]); }
        protected AbstractPreferences childSpi(String n) { return children.computeIfAbsent(n, x -> new MemoryPreferences(this, x)); }
        protected void syncSpi() { }
        protected void flushSpi() { }
    }

    public static void main(String[] args) {
        try {
            run(args);
            System.out.flush();
            System.exit(0);
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(2);
        }
    }

    static void run(String[] args) throws Exception {
        File in = new File(args[0]);
        String out = args[1];
        Map<String, String> o = new HashMap<>();
        for (int i = 2; i < args.length; i++) {
            String[] kv = args[i].split("=", 2);
            o.put(kv[0], kv[1]);
        }
        ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
        pc.newProject();
        Workspace ws = pc.getCurrentWorkspace();
        String nome = new File(out).getName();
        pc.renameProject(pc.getCurrentProject(), nome);
        pc.renameWorkspace(ws, nome);
        ImportController ic = Lookup.getDefault().lookup(ImportController.class);
        Container c = ic.importFile(in);
        c.getLoader().setEdgeDefault(EdgeDirectionDefault.UNDIRECTED);
        c.getLoader().setAllowAutoNode(false);
        ic.process(c, new DefaultProcessor(), ws);
        GraphModel gm = Lookup.getDefault().lookup(GraphController.class).getGraphModel(ws);
        Graph g = gm.getUndirectedGraph();
        System.out.println("importado: nos=" + g.getNodeCount() + " arestas=" + g.getEdgeCount());

        if ("fa2".equals(o.getOrDefault("layout", "fixo"))) {
            ForceAtlas2 fa2 = new ForceAtlas2Builder().buildLayout();
            fa2.setGraphModel(gm);
            fa2.resetPropertiesValues();
            fa2.setBarnesHutOptimize(true);
            fa2.setBarnesHutTheta(Double.parseDouble(o.getOrDefault("theta", "1.2")));
            fa2.setLinLogMode(Boolean.parseBoolean(o.getOrDefault("linlog", "true")));
            fa2.setScalingRatio(Double.parseDouble(o.getOrDefault("scaling", "2.0")));
            fa2.setGravity(Double.parseDouble(o.getOrDefault("gravity", "1.0")));
            fa2.setStrongGravityMode(Boolean.parseBoolean(o.getOrDefault("strong", "false")));
            fa2.setOutboundAttractionDistribution(Boolean.parseBoolean(o.getOrDefault("dissuade", "true")));
            fa2.setEdgeWeightInfluence(Double.parseDouble(o.getOrDefault("weightInfluence", "1.0")));
            fa2.setThreadsCount(Integer.parseInt(o.getOrDefault("threads", "4")));
            fa2.setAdjustSizes(false);
            int iter = Integer.parseInt(o.getOrDefault("iter", "1500"));
            fa2.initAlgo();
            long t0 = System.currentTimeMillis();
            for (int i = 0; i < iter && fa2.canAlgo(); i++) {
                fa2.goAlgo();
                if (i % 100 == 0) System.out.println("fa2 iter " + i + " t=" + (System.currentTimeMillis() - t0) / 1000 + "s");
            }
            int fine = Integer.parseInt(o.getOrDefault("iterAjuste", "0"));
            if (fine > 0) {
                fa2.setAdjustSizes(true);
                for (int i = 0; i < fine && fa2.canAlgo(); i++) fa2.goAlgo();
            }
            fa2.endAlgo();
            if (Boolean.parseBoolean(o.getOrDefault("noverlap", "false"))) {
                NoverlapLayout nv = (NoverlapLayout) new NoverlapLayoutBuilder().buildLayout();
                nv.setGraphModel(gm);
                nv.resetPropertiesValues();
                nv.setMargin(Double.parseDouble(o.getOrDefault("noverlapMargin", "1.0")));
                nv.initAlgo();
                for (int i = 0; i < 200 && nv.canAlgo(); i++) nv.goAlgo();
                nv.endAlgo();
            }
        }

        PreviewController prc = Lookup.getDefault().lookup(PreviewController.class);
        PreviewModel pm = prc.getModel(ws);
        var p = pm.getProperties();
        p.putValue(PreviewProperty.BACKGROUND_COLOR, Color.WHITE);
        p.putValue(PreviewProperty.DIRECTED, false);
        p.putValue(PreviewProperty.ARROW_SIZE, 0f);
        p.putValue(PreviewProperty.EDGE_CURVED, false);
        p.putValue(PreviewProperty.EDGE_RESCALE_WEIGHT, Boolean.parseBoolean(o.getOrDefault("rescale", "false")));
        p.putValue(PreviewProperty.EDGE_THICKNESS, Float.parseFloat(o.getOrDefault("edgeThickness", "0.4")));
        p.putValue(PreviewProperty.EDGE_OPACITY, Float.parseFloat(o.getOrDefault("edgeOpacity", "25")));
        p.putValue(PreviewProperty.EDGE_COLOR, new EdgeColor(EdgeColor.Mode.ORIGINAL));
        p.putValue(PreviewProperty.NODE_BORDER_WIDTH, Float.parseFloat(o.getOrDefault("border", "0.3")));
        p.putValue(PreviewProperty.NODE_OPACITY, 100f);
        boolean labels = Boolean.parseBoolean(o.getOrDefault("labels", "true"));
        p.putValue(PreviewProperty.SHOW_NODE_LABELS, labels);
        p.putValue(PreviewProperty.NODE_LABEL_PROPORTIONAL_SIZE, false);
        p.putValue(PreviewProperty.NODE_LABEL_FONT, new Font("Arial", Font.PLAIN, Integer.parseInt(o.getOrDefault("labelSize", "10"))));
        p.putValue(PreviewProperty.NODE_LABEL_OUTLINE_SIZE, 2f);
        p.putValue(PreviewProperty.NODE_LABEL_OUTLINE_COLOR, new org.gephi.preview.types.DependantColor(Color.WHITE));
        p.putValue(PreviewProperty.SHOW_EDGE_LABELS, Boolean.parseBoolean(o.getOrDefault("edgeLabels", "false")));
        p.putValue(PreviewProperty.MARGIN, 4f);
        prc.refreshPreview(ws);

        ExportController ec = Lookup.getDefault().lookup(ExportController.class);
        // Coordenadas finais calculadas pelo Gephi (lidas na composição).
        try (java.io.PrintWriter w = new java.io.PrintWriter(new File(out + "_posicoes.csv"), "UTF-8")) {
            w.println("Id,x,y,size");
            for (Node n : g.getNodes()) w.println(n.getId() + "," + n.x() + "," + n.y() + "," + n.size());
        }
        ec.exportFile(new File(out + "_preview.svg"));
        // O PDFExporter do Gephi 0.10.1 é quadrático no nº de arestas (busca linear de ExtGState por aresta):
        // inviável para o grafo completo; nesse caso use pdf=false e o PDF da composição.
        if (Boolean.parseBoolean(o.getOrDefault("pdf", "true"))) {
            PDFExporter pdf = (PDFExporter) ec.getExporter("pdf");
            pdf.setWorkspace(ws);
            ec.exportFile(new File(out + "_preview.pdf"), pdf);
        }
        // PNG do Gephi bloqueia em modo headless; o PNG final é gerado na composição.
        int px = Integer.parseInt(o.getOrDefault("png", "0"));
        if (px > 0) {
            PNGExporter png = (PNGExporter) ec.getExporter("png");
            png.setWorkspace(ws);
            png.setWidth(px);
            png.setHeight(px);
            png.setTransparentBackground(false);
            ec.exportFile(new File(out + "_preview.png"), png);
        }
        File proj = new File(out + ".gephi");
        // Gephi 0.10.1: se o destino não existe, a limpeza do SaveTask apaga o próprio arquivo gravado.
        // Com o destino pré-existente, ele grava num temporário e renomeia (comportamento da interface).
        java.nio.file.Files.deleteIfExists(proj.toPath());
        proj.createNewFile();
        org.gephi.project.io.SaveTask save = new org.gephi.project.io.SaveTask(pc.getCurrentProject(), proj);
        boolean r = save.run();
        System.out.println("projeto " + proj.getAbsolutePath() + " savetask=" + r + " existe=" + proj.exists() + " bytes=" + proj.length());
        if (!proj.exists()) throw new IllegalStateException("projeto .gephi nao foi gravado");
        System.out.println("ok " + out);
    }
}
