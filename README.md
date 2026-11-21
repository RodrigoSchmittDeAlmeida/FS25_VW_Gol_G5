# FS25_VW_Gol_G5 — Volkswagen Gol G5 para Farming Simulator 25

Mod completo do **Volkswagen Gol G5 (2009–2012)** desenvolvido para **Farming Simulator 25 (FS25)**, construído a partir do modelo 3D original do portfólio [RodrigoSchmittDeAlmeida.github.io](https://github.com/RodrigoSchmittDeAlmeida/RodrigoSchmittDeAlmeida.github.io) e com calibração de motorizações inspirada nos mapas de injeção eletrônica estudados no projeto **`leitorECU`**.

---

## Visão Geral do Mod

- **Veículo:** Volkswagen Gol G5 Hatch 4 portas (`category="carsBikes"`, `type="car"`)
- **Marca:** Volkswagen (`brand_vw.dds`)
- **Capacidade do Tanque:** `55 L` (padrão real do VW Gol G5)
- **Tração:** Dianteira (FWD), diferencial dianteiro e câmbio MQ200 de 5 marchas + ré
- **Recursos Funcionais no FS25:**
  - Modelo 3D detalhado (exterior, interior, painel, suspensão e cubos de roda) em **I3D v1.6**
  - Volante funcional (`steeringWheel`) alinhado no eixo real da coluna de direção (`65°`) + alvos IK de mãos e pés para o motorista
  - Iluminação completa: faróis baixos/altos (`defaultLights` + `realLights`), luzes de freio (`brakeLights`) e luzes de ré (`reverseLights`)
  - Engate traseiro leve (`trailerCar`) para carretinhas e pequenos reboques
  - Texturas **DDS DXT5** com cadeia completa de mipmaps (`icon_golG5.dds`, `store_golG5.dds`, `brand_vw.dds` e 6 texturas de lanternas/faróis)
  - Localização completa em Português Brasileiro (`l10n_br.xml`) e Inglês (`l10n_en.xml`)
  - Compatível com Multiplayer (`supported="true"`)

---

## Motorizações EA111 VHT

O mod oferece 3 opções de motor na loja do jogo:

| Configuração | Potência | Velocidade Máx. | Preço Adicional | Observações |
| :--- | :---: | :---: | :---: | :--- |
| **1.0 VHT** *(Padrão)* | **76 cv** (`76 hp`) | `165 km/h` | `€ 0` | Motor EA111 1.0 VHT original, faixa até `6.200 rpm` |
| **1.6 VHT** | **104 cv** (`104 hp`) | `190 km/h` | `€ 3.800` | Motor EA111 1.6 VHT original, torque reforçado em médias rotações |
| **1.6 VHT Stage 1 ME7.5.30** | **125 cv** (`125 hp`) | `208 km/h` | `€ 6.500` | Remap Stage 1 da central Bosch ME7.5.30, corte estendido para `6.500 rpm` e freios aprimorados |

---

## Paleta de Cores Originais Volkswagen (`baseColorConfigurations`)

A lataria utiliza `materialSlotName="carPaint"` no `vehicleShader.xml` com 6 cores originais VW:

| Nome (PT-BR / EN) | RGB Linear (XML) | Acabamento | Preço |
| :--- | :--- | :--- | :---: |
| **Preto Ninja** / *Ninja Black* | `0.015 0.016 0.020` | Sólido *(Padrão)* | `€ 0` |
| **Branco Cristal** / *Crystal White* | `0.90 0.91 0.92` | Sólido | `€ 500` |
| **Prata Sirius** / *Sirius Silver* | `0.58 0.60 0.63` | Metálico | `€ 750` |
| **Cinza Spectrus** / *Spectrus Gray* | `0.20 0.22 0.24` | Metálico | `€ 750` |
| **Vermelho Flash** / *Flash Red* | `0.68 0.02 0.02` | Sólido | `€ 600` |
| **Azul Boreal** / *Boreal Blue* | `0.03 0.14 0.36` | Metálico | `€ 750` |

---

## Instalação no Farming Simulator 25

1. Baixe ou gere o pacote `dist/FS25_VW_Gol_G5.zip`.
2. Copie o arquivo **`FS25_VW_Gol_G5.zip`** diretamente para a pasta `mods` do Farming Simulator 25:
   - **Windows:** `Documentos\My Games\FarmingSimulator2025\mods\`
   - **Steam Deck / Linux (Proton):** `~/.local/share/Steam/steamapps/compatdata/2300320/pfx/drive_c/users/steamuser/Documents/My Games/FarmingSimulator2025/mods/`
3. Abra o jogo, ative o mod **Volkswagen Gol G5** na tela de seleção de mods do seu save e compre o carro na loja em **Veículos > Carros e Motos** (`carsBikes`).

> **Dica opcional de otimização (`.i3d.shapes`) no GIANTS Editor 10:**
> O arquivo `golG5.i3d` é distribuído em XML I3D v1.6 puro (legível e autocontido). Caso deseje reduzir o tempo de carregamento e o tamanho do arquivo em disco para uso em servidores dedicados, abra `FS25_VW_Gol_G5/vehicles/vwGolG5/golG5.i3d` no **GIANTS Editor 10** e salve novamente (`File > Save`) para que o editor gere automaticamente o binário complementar `golG5.i3d.shapes`.

---

## Scripts e Ferramentas (`tools/`)

O repositório inclui todo o pipeline automatizado em Python (`Python 3.11+`, `numpy`, `Pillow`, `lxml`, `xmlschema`):

- [`tools/generate_dds_assets.py`](tools/generate_dds_assets.py): Lê os assets de origem em `source_assets/` (`gol_g5.glb` e `gol_g5_preview.png`, ignorados pelo Git) e gera todas as 9 texturas **DDS DXT5** com cadeia completa de mipmaps (`icon_golG5.dds`, `store_golG5.dds`, `brand_vw.dds` e `textures/*.dds`).
- [`tools/convert_glb_to_i3d.py`](tools/convert_glb_to_i3d.py): Converte `source_assets/gol_g5.glb` para o sistema de coordenadas do GIANTS Engine (`+Y` cima, `+Z` frente, `+X` esquerda, pneus apoiados em `Y = 0.000`), centraliza rodas e volante, emite `golG5.i3d` (v1.6) e gera `tools/i3d_node_map.json`.
- [`tools/build_mod_zip.py`](tools/build_mod_zip.py): Empacota `FS25_VW_Gol_G5/` em `dist/FS25_VW_Gol_G5.zip` com `modDesc.xml` na raiz do arquivo.
- [`tools/validate_mod.py`](tools/validate_mod.py): Executa a suíte completa de **7 validações**:
  1. Validação oficial contra `modDesc.xsd` e `vehicle.xsd` da GIANTS Software
  2. Paridade de chaves e referências `$l10n_*` entre `l10n_en.xml` e `l10n_br.xml`
  3. Integridade estrutural do `golG5.i3d` v1.6 (`fileId`, `materialIds`, `shapeId`, `carPaint`, contagens)
  4. Resolução 100% de todos os `<i3dMapping>` entre `golG5.xml` e `golG5.i3d`
  5. Inspeção binária dos 9 arquivos `.dds` (cabeçalho `DDS `, `DXT5`, dimensões e mipmaps)
  6. Integridade do `dist/FS25_VW_Gol_G5.zip` e presença de todos os assets referenciados
  7. Conferência da política do `.gitignore`

### Como recompilar e validar localmente

```bash
pip install numpy Pillow lxml xmlschema
python tools/generate_dds_assets.py
python tools/convert_glb_to_i3d.py
python tools/build_mod_zip.py
python tools/validate_mod.py
```

---

## Créditos

- **Autor do Mod & Portfólio 3D:** [Rodrigo Schmitt de Almeida](https://github.com/RodrigoSchmittDeAlmeida) — modelo 3D base originado de [`RodrigoSchmittDeAlmeida.github.io`](https://github.com/RodrigoSchmittDeAlmeida/RodrigoSchmittDeAlmeida.github.io).
- **Referência Técnica ECU (1.6 VHT Stage 1 ME7.5.30):** Baseado nas curvas de torque e parâmetros de calibração do projeto **`leitorECU`** (Bosch ME7.5.30).
- **Schemas Oficiais:** [GIANTS Developer Network (GDN)](https://validation.gdn.giants-software.com/) (`modDesc.xsd` / `vehicle.xsd`).
