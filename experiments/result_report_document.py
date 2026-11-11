"""Create a self-contained Word report and a portable HTML results viewer.

Uses only Python standard-library OOXML generation; no Word automation needed.
"""
import base64,csv,html,json,math,os,tempfile,time,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
def load(p):return json.loads((HERE/p).read_text(encoding='utf-8'))
def num(x,d=4):return '—' if x is None else ('{:.%dg}'%d).format(x)
def pct(x):return '未覆盖' if x is None else '{:.3f}%'.format(100*x)
def compact(r):return '未覆盖' if not r or not r.get('covered',True) else '{} / {:.2f}×'.format(pct(r['relative_error']),r['cr'])

class Report:
    def __init__(self,figure_dir=None):
        self.body=[];self.html=[];self.images=[]
        self.figure_dir=Path(figure_dir) if figure_dir is not None else HERE/'figures'
    def p(self,text='',style=None):
        prop='<w:pPr>'+('<w:pStyle w:val="'+style+'"/>' if style else '')+'</w:pPr>'
        self.body.append('<w:p>'+prop+'<w:r><w:t xml:space="preserve">'+escape(str(text))+'</w:t></w:r></w:p>')
        tag='h1' if style=='Title' else 'h2' if style=='Heading1' else 'h3' if style=='Heading2' else 'p'
        self.html.append('<'+tag+'>'+html.escape(str(text))+'</'+tag+'>')
    def h(self,text,level=1):self.p(text,'Heading'+str(level))
    def table(self,headers,rows,small=False):
        rows=list(rows);n=len(headers);width=10100//max(n,1);font=16 if small or n>=7 else 18
        s=['<w:tbl><w:tblPr><w:tblW w:w="10100" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>'+''.join('<w:'+k+' w:val="single" w:sz="4" w:color="CCD6DF"/>' for k in ['top','left','bottom','right','insideH','insideV'])+'</w:tblBorders><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'+('<w:gridCol w:w="'+str(width)+'"/>')*n+'</w:tblGrid>']
        for i,row in enumerate([headers]+rows):
            s.append('<w:tr>'+('<w:trPr><w:tblHeader/></w:trPr>' if i==0 else ''))
            for value in row:
                cellprop='<w:tcPr><w:tcW w:w="'+str(width)+'" w:type="dxa"/>'+('<w:shd w:fill="173D61"/>' if i==0 else '<w:shd w:fill="F1F5F8"/>' if i%2==0 else '')+'</w:tcPr>'
                rprop='<w:rPr><w:sz w:val="'+str(font)+'"/>'+('<w:b/><w:color w:val="FFFFFF"/>' if i==0 else '')+'</w:rPr>'
                s.append('<w:tc>'+cellprop+'<w:p><w:pPr><w:spacing w:after="20" w:line="240" w:lineRule="auto"/></w:pPr><w:r>'+rprop+'<w:t xml:space="preserve">'+escape(str(value))+'</w:t></w:r></w:p></w:tc>')
            s.append('</w:tr>')
        s.append('</w:tbl>');self.body.append(''.join(s));self.p()
        self.html.append('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>')
    def figure(self,name,caption):
        from struct import unpack
        p=self.figure_dir/(name+'.png');data=p.read_bytes();w,h=unpack('>II',data[16:24]);cx=6300000;cy=int(cx*h/w);idx=len(self.images)+1;rid='rIdImg'+str(idx)
        self.images.append((rid,p,data))
        xml='<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="'+str(cx)+'" cy="'+str(cy)+'"/><wp:docPr id="'+str(idx)+'" name="Figure '+str(idx)+'" descr="'+escape(caption)+'"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="'+name+'.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="'+rid+'"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="'+str(cx)+'" cy="'+str(cy)+'"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
        self.body.append(xml);self.p(caption,'Caption')
        self.html.append('<figure><img src="data:image/png;base64,'+base64.b64encode(data).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
    def pagebreak(self):self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    def save(self,path,html_path=None):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        html_path=Path(html_path) if html_path is not None else OUT/'QTJoint_实验结果展示.html'
        html_path.parent.mkdir(parents=True,exist_ok=True)
        ns='xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
        section='<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="900" w:right="850" w:bottom="900" w:left="850" w:header="400" w:footer="450"/></w:sectPr>'
        document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document '+ns+'><w:body>'+''.join(self.body)+section+'</w:body></w:document>'
        styles='''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="310" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:before="500" w:after="350"/></w:pPr><w:rPr><w:b/><w:color w:val="173D61"/><w:sz w:val="44"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="280" w:after="140"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="173D61"/><w:sz w:val="30"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="90"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="245D81"/><w:sz w:val="25"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:sz w:val="18"/><w:color w:val="546472"/></w:rPr></w:style></w:styles>'''
        relns='http://schemas.openxmlformats.org/package/2006/relationships';base='http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        rels='<?xml version="1.0"?><Relationships xmlns="'+relns+'"><Relationship Id="rIdStyles" Type="'+base+'styles" Target="styles.xml"/><Relationship Id="rIdFooter" Type="'+base+'footer" Target="footer1.xml"/>'+''.join('<Relationship Id="'+rid+'" Type="'+base+'image" Target="media/'+p.name+'"/>' for rid,p,data in self.images)+'</Relationships>'
        content='<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'
        rootrels='<?xml version="1.0"?><Relationships xmlns="'+relns+'"><Relationship Id="rId1" Type="'+base+'officeDocument" Target="word/document.xml"/></Relationships>'
        footer='<?xml version="1.0"?><w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:t>QT-Joint 实验测试评估 · </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'
        handle,temp_path=tempfile.mkstemp(prefix=path.stem+'_',suffix='.docx',dir=path.parent)
        os.close(handle)
        try:
            with zipfile.ZipFile(temp_path,'w',zipfile.ZIP_DEFLATED) as z:
                for name,text in [('[Content_Types].xml',content),('_rels/.rels',rootrels),('word/document.xml',document),('word/styles.xml',styles),('word/_rels/document.xml.rels',rels),('word/footer1.xml',footer)]:z.writestr(name,text.encode('utf-8'))
                for rid,p,data in self.images:z.writestr('word/media/'+p.name,data)
            with zipfile.ZipFile(temp_path) as z:
                for n in z.namelist():
                    if n.endswith(('.xml','.rels')):ET.fromstring(z.read(n))
            os.replace(temp_path,path)
        finally:
            if os.path.exists(temp_path):os.unlink(temp_path)
        style='''body{font-family:"Microsoft YaHei",Arial,sans-serif;color:#203345;background:#eef3f7;margin:0}main{max-width:1160px;margin:30px auto;padding:42px;background:white;border-radius:14px}h1,h2,h3{color:#173d61}h2{margin-top:42px;border-bottom:2px solid #e5edf3;padding-bottom:10px}p{line-height:1.8}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}th{background:#173d61;color:white}td,th{padding:8px;border:1px solid #d4dfe8;text-align:left}tr:nth-child(even){background:#f1f5f8}img{max-width:100%}figure{margin:24px 0}figcaption{font-size:13px;color:#657685}input{font:inherit;padding:10px;width:95%;margin:10px 0;border:1px solid #a9bdcc;border-radius:5px}@media print{body{background:white}main{margin:0;padding:0}figure{break-inside:avoid}}'''
        js='''document.querySelectorAll('table').forEach(t=>{if(t.rows.length>20){let i=document.createElement('input');i.placeholder='筛选本表：数据集、方法、运行编号或指标';i.addEventListener('input',()=>{let q=i.value.toLowerCase();Array.from(t.tBodies[0].rows).forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});t.parentNode.insertBefore(i,t)}})'''
        html_path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QT-Joint 实验测试评估</title><style>'+style+'</style><main>'+''.join(self.html)+'</main><script>'+js+'</script></html>',encoding='utf-8')
